import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def batched(items: List[Dict], n: int) -> Iterable[List[Dict]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    existing = {i["name"] for i in pc.list_indexes()}
    if index_name in existing:
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def chunk_path_from_strategy(strategy: str) -> Path:
    base = Path(__file__).resolve().parent
    # utils/ is one level below repo root in this project.
    c1 = base.parent / "rag_artifacts" / "chunks"
    if c1.exists():
        return c1 / f"chunks_{strategy}.jsonl"

    # Fallback for layouts where code is nested under src/
    c2 = base.parent.parent / "rag_artifacts" / "chunks"
    return c2 / f"chunks_{strategy}.jsonl"


def build_metadata(row: Dict, include_text: bool = True) -> Dict:
    metadata = {
        "doc_id": row.get("doc_id", ""),
        "source_file": row.get("source_file", ""),
        "title": row.get("title", ""),
        "category": row.get("category", ""),
        "strategy": row.get("strategy", ""),
        "chunk_index": int(row.get("chunk_index", 0)),
        "chunk_word_count": int(row.get("chunk_word_count", 0)),
    }
    if include_text:
        metadata["text"] = row.get("text", "")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsert Urdu RAG chunks to Pinecone.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed",
        choices=["fixed", "recursive", "sentence"],
        help="Chunking strategy whose JSONL should be indexed.",
    )
    parser.add_argument(
        "--chunk-file",
        type=str,
        default="",
        help="Optional custom JSONL path. If omitted, inferred from --strategy.",
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
        help="Pinecone namespace. Defaults to the strategy name.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model for both docs and queries.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding/upsert batch size.",
    )
    parser.add_argument(
        "--cloud",
        type=str,
        default="aws",
        help="Pinecone serverless cloud.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="Pinecone serverless region.",
    )
    parser.add_argument(
        "--include-text-metadata",
        action="store_true",
        help="Store chunk text in Pinecone metadata for direct retrieval and citations.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment or .env file.")

    namespace = args.namespace.strip() or args.strategy
    chunk_file = (
        Path(args.chunk_file)
        if args.chunk_file
        else chunk_path_from_strategy(args.strategy)
    )
    if not chunk_file.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

    print(f"Loading chunks from {chunk_file} ...")
    rows = read_jsonl(chunk_file)
    if not rows:
        raise RuntimeError("Chunk file is empty.")

    model = SentenceTransformer(args.embed_model)
    test_vec = model.encode(["تجرباتی جملہ"], normalize_embeddings=True)[0]
    dimension = len(test_vec)

    pc = Pinecone(api_key=api_key)
    ensure_index(
        pc,
        index_name=args.index_name,
        dimension=dimension,
        metric="cosine",
        cloud=args.cloud,
        region=args.region,
    )
    index = pc.Index(args.index_name)

    print(
        f"Indexing {len(rows)} chunks | strategy={args.strategy} | namespace={namespace} | dim={dimension}"
    )

    total = 0
    start = time.time()
    for batch in batched(rows, args.batch_size):
        texts = [r.get("text", "") for r in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        vectors = []
        for row, emb in zip(batch, embeddings):
            vector_id = row.get("chunk_id")
            if not vector_id:
                continue
            vectors.append(
                {
                    "id": vector_id,
                    "values": emb,
                    "metadata": build_metadata(
                        row, include_text=args.include_text_metadata
                    ),
                }
            )

        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
            total += len(vectors)
            print(f"Upserted {total}/{len(rows)}")

    elapsed = round(time.time() - start, 2)
    print(
        f"Done. Upserted {total} vectors in {elapsed}s into index={args.index_name}, namespace={namespace}"
    )


if __name__ == "__main__":
    main()
