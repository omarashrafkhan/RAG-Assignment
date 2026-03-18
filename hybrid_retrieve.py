import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_urdu(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"\S+", text, flags=re.UNICODE)


def build_bm25(chunks: List[Dict]) -> BM25Okapi:
    tokenized = [tokenize_urdu(c.get("text", "")) for c in chunks]
    return BM25Okapi(tokenized)


def bm25_search(
    bm25: BM25Okapi, chunks: List[Dict], query: str, top_k: int
) -> List[Dict]:
    tokens = tokenize_urdu(query)
    scores = bm25.get_scores(tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]
    out: List[Dict] = []
    for rank, idx in enumerate(ranked_idx, start=1):
        c = chunks[idx]
        out.append(
            {
                "chunk_id": c["chunk_id"],
                "rank": rank,
                "score": float(scores[idx]),
                "text": c.get("text", ""),
                "source_file": c.get("source_file", ""),
                "title": c.get("title", ""),
                "category": c.get("category", ""),
            }
        )
    return out


def semantic_search(
    query: str,
    embedder: SentenceTransformer,
    index_name: str,
    namespace: str,
    top_k: int,
) -> List[Dict]:
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment or .env file.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    qvec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    response = index.query(
        vector=qvec, top_k=top_k, namespace=namespace, include_metadata=True
    )

    out: List[Dict] = []
    matches = (
        response.get("matches", []) if isinstance(response, dict) else response.matches
    )
    for rank, m in enumerate(matches, start=1):
        md = (
            m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
        )
        score = m.get("score", 0.0) if isinstance(m, dict) else getattr(m, "score", 0.0)
        chunk_id = m.get("id", "") if isinstance(m, dict) else getattr(m, "id", "")

        out.append(
            {
                "chunk_id": chunk_id,
                "rank": rank,
                "score": float(score),
                "text": md.get("text", ""),
                "source_file": md.get("source_file", ""),
                "title": md.get("title", ""),
                "category": md.get("category", ""),
            }
        )
    return out


def reciprocal_rank_fusion(
    bm25_hits: List[Dict], semantic_hits: List[Dict], k: int = 60
) -> List[Dict]:
    fused: Dict[str, Dict] = {}

    for hit in bm25_hits:
        cid = hit["chunk_id"]
        fused.setdefault(
            cid, {"chunk_id": cid, "rrf": 0.0, "bm25_rank": None, "semantic_rank": None}
        )
        fused[cid]["rrf"] += 1.0 / (k + hit["rank"])
        fused[cid]["bm25_rank"] = hit["rank"]
        if hit.get("text"):
            fused[cid]["text"] = hit["text"]
            fused[cid]["title"] = hit.get("title", "")
            fused[cid]["source_file"] = hit.get("source_file", "")
            fused[cid]["category"] = hit.get("category", "")

    for hit in semantic_hits:
        cid = hit["chunk_id"]
        fused.setdefault(
            cid, {"chunk_id": cid, "rrf": 0.0, "bm25_rank": None, "semantic_rank": None}
        )
        fused[cid]["rrf"] += 1.0 / (k + hit["rank"])
        fused[cid]["semantic_rank"] = hit["rank"]
        if hit.get("text"):
            fused[cid]["text"] = hit["text"]
            fused[cid]["title"] = hit.get("title", "")
            fused[cid]["source_file"] = hit.get("source_file", "")
            fused[cid]["category"] = hit.get("category", "")

    results = list(fused.values())
    results.sort(key=lambda x: x["rrf"], reverse=True)
    return results


def rerank(
    query: str,
    hits: List[Dict],
    reranker_model: str,
    top_k: int,
) -> List[Dict]:
    if not hits:
        return []

    model = CrossEncoder(reranker_model)
    pairs = [(query, h.get("text", "")) for h in hits]
    scores = model.predict(pairs)

    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)

    hits.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return hits[:top_k]


def chunk_path_from_strategy(strategy: str) -> Path:
    base = Path("rag_artifacts") / "chunks"
    return base / f"chunks_{strategy}.jsonl"


def fill_missing_text_from_local(
    hits: List[Dict], local_by_id: Dict[str, Dict]
) -> None:
    for h in hits:
        if h.get("text"):
            continue
        local = local_by_id.get(h["chunk_id"], {})
        h["text"] = local.get("text", "")
        h["title"] = local.get("title", "")
        h["source_file"] = local.get("source_file", "")
        h["category"] = local.get("category", "")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Hybrid retrieval for Urdu RAG (BM25 + Semantic + RRF)."
    )
    parser.add_argument("--query", type=str, required=True, help="User query in Urdu.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed",
        choices=["fixed", "recursive", "sentence"],
        help="Chunking strategy used in retrieval.",
    )
    parser.add_argument(
        "--chunk-file",
        type=str,
        default="",
        help="Optional custom chunk JSONL. If empty, inferred from --strategy.",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="urdu-medical-rag",
        help="Pinecone index name.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="",
        help="Pinecone namespace. Defaults to strategy.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Query embedding model.",
    )
    parser.add_argument("--bm25-top-k", type=int, default=30)
    parser.add_argument("--semantic-top-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Run retrieval only with BM25 (no Pinecone semantic retrieval).",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranking on fused candidates.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Multilingual reranker model.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save full retrieval output JSON.",
    )
    args = parser.parse_args()

    namespace = args.namespace.strip() or args.strategy
    chunk_file = (
        Path(args.chunk_file)
        if args.chunk_file
        else chunk_path_from_strategy(args.strategy)
    )
    if not chunk_file.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

    chunks = read_jsonl(chunk_file)
    local_by_id = {c["chunk_id"]: c for c in chunks}

    bm25 = build_bm25(chunks)
    bm25_hits = bm25_search(bm25, chunks, args.query, top_k=args.bm25_top_k)

    semantic_hits: List[Dict] = []
    if not args.bm25_only and args.semantic_top_k > 0:
        embedder = SentenceTransformer(args.embed_model)
        semantic_hits = semantic_search(
            query=args.query,
            embedder=embedder,
            index_name=args.index_name,
            namespace=namespace,
            top_k=args.semantic_top_k,
        )

    fused = reciprocal_rank_fusion(bm25_hits, semantic_hits, k=args.rrf_k)
    fill_missing_text_from_local(fused, local_by_id)
    fused = fused[: max(args.final_top_k * 3, args.final_top_k)]

    if args.use_reranker:
        final_hits = rerank(
            query=args.query,
            hits=fused,
            reranker_model=args.reranker_model,
            top_k=args.final_top_k,
        )
    else:
        final_hits = fused[: args.final_top_k]

    payload = {
        "query": args.query,
        "strategy": args.strategy,
        "namespace": namespace,
        "index_name": args.index_name,
        "bm25_top_k": args.bm25_top_k,
        "semantic_top_k": args.semantic_top_k,
        "bm25_only": args.bm25_only,
        "final_top_k": args.final_top_k,
        "used_reranker": args.use_reranker,
        "results": final_hits,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
