import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from hybrid_retrieve import (
    bm25_search,
    build_bm25,
    chunk_path_from_strategy,
    fill_missing_text_from_local,
    read_jsonl,
    reciprocal_rank_fusion,
    rerank,
    semantic_search,
)


def retrieve_chunks(
    args: argparse.Namespace, embedder: Optional[SentenceTransformer] = None
) -> List[Dict]:
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
        if embedder is None:
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

    return final_hits


def build_prompt(query: str, hits: List[Dict], max_context_chars: int = 10000) -> str:
    context_blocks: List[str] = []
    total_chars = 0

    for i, h in enumerate(hits, start=1):
        text = (h.get("text") or "").strip()
        if not text:
            continue

        block = (
            f"[{i}]"
            f"\nعنوان: {h.get('title', '')}"
            f"\nماخذ: {h.get('source_file', '')}"
            f"\nمتن: {text}\n"
        )
        if total_chars + len(block) > max_context_chars:
            break
        context_blocks.append(block)
        total_chars += len(block)

    joined_context = "\n".join(context_blocks)

    return f"""
آپ ایک طبی سوال جواب نظام ہیں۔
صرف دیے گئے سیاق (context) سے جواب دیں۔
اگر سیاق میں جواب موجود نہ ہو تو صاف لکھیں: "دی گئی معلومات میں واضح جواب موجود نہیں"۔
جواب اردو میں دیں، مختصر اور درست رکھیں، اور ہر اہم دعوے کے ساتھ حوالہ [1] [2] جیسی شکل میں دیں۔

سوال:
{query}

سیاق:
{joined_context}

متوقع جواب (اردو + حوالہ جات):
""".strip()


def call_github_models(
    prompt: str,
    model_id: str,
    github_token: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, Dict]:
    client = OpenAI(api_key=github_token, base_url="https://models.github.ai/inference")

    errors: List[str] = []
    backoffs = [1.0, 2.0, 4.0]

    for delay in backoffs:
        try:
            chat = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if chat and chat.choices:
                text = (chat.choices[0].message.content or "").strip()
                if text:
                    return text, {"raw": "chat.completions"}
        except Exception as exc:
            errors.append(f"chat.completions: {str(exc)[:220]}")

        time.sleep(delay)

    raise RuntimeError(
        "GitHub Models generation failed after retries. "
        f"Model={model_id}. Recent errors={errors[-4:]}"
    )


def parse_model_list(raw: str) -> List[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def call_github_models_with_fallback(
    prompt: str,
    primary_model: str,
    fallback_models: List[str],
    github_token: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, Dict, str]:
    # Keep GPT-4 mini as reliable fallback (widely available on GitHub Models).
    candidates = [primary_model] + fallback_models + ["openai/gpt-4.1-mini"]
    seen = set()
    ordered: List[str] = []
    for m in candidates:
        if m and m not in seen:
            ordered.append(m)
            seen.add(m)

    errors: List[str] = []
    for model_id in ordered:
        try:
            text, meta = call_github_models(
                prompt=prompt,
                model_id=model_id,
                github_token=github_token,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return text, meta, model_id
        except Exception as exc:
            errors.append(f"{model_id}: {str(exc)[:200]}")

    raise RuntimeError(
        f"All candidate models failed. Tried={ordered}. Errors={errors[-4:]}"
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate Urdu RAG answer using Hybrid Retrieval + HF Inference API."
    )
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed",
        choices=["fixed", "recursive", "sentence"],
    )
    parser.add_argument("--chunk-file", type=str, default="")
    parser.add_argument("--index-name", type=str, default="urdu-medical-rag")
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--bm25-top-k", type=int, default=30)
    parser.add_argument("--semantic-top-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default="openai/gpt-4.1-mini",
        help="GitHub Models model ID for generation.",
    )
    parser.add_argument(
        "--generation-fallback-models",
        type=str,
        default="",
        help="Comma-separated fallback model IDs used when primary model fails.",
    )
    parser.add_argument("--max-context-chars", type=int, default=10000)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    if not github_token:
        raise RuntimeError("Missing GITHUB_TOKEN in environment or .env file.")

    shared_embedder = None
    if not args.bm25_only and args.semantic_top_k > 0:
        shared_embedder = SentenceTransformer(args.embed_model)

    hits = retrieve_chunks(args, embedder=shared_embedder)
    prompt = build_prompt(args.query, hits, max_context_chars=args.max_context_chars)
    answer, meta, used_model = call_github_models_with_fallback(
        prompt=prompt,
        primary_model=args.generation_model,
        fallback_models=parse_model_list(args.generation_fallback_models),
        github_token=github_token,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    payload = {
        "query": args.query,
        "strategy": args.strategy,
        "index_name": args.index_name,
        "namespace": args.namespace.strip() or args.strategy,
        "bm25_only": args.bm25_only,
        "used_reranker": args.use_reranker,
        "generation_model": args.generation_model,
        "used_generation_model": used_model,
        "answer": answer,
        "retrieved_context": [
            {
                "citation_id": i + 1,
                "chunk_id": h.get("chunk_id", ""),
                "title": h.get("title", ""),
                "source_file": h.get("source_file", ""),
                "category": h.get("category", ""),
                "rrf": h.get("rrf", None),
                "bm25_rank": h.get("bm25_rank", None),
                "semantic_rank": h.get("semantic_rank", None),
                "rerank_score": h.get("rerank_score", None),
                "text": h.get("text", ""),
            }
            for i, h in enumerate(hits)
        ],
        "hf_meta": meta,
    }

    json_str = json.dumps(payload, ensure_ascii=False, indent=2)
    import sys

    sys.stdout.buffer.write(json_str.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
