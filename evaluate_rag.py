import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from generate_answer import (
    build_prompt,
    call_github_models_with_fallback,
    parse_model_list,
    retrieve_chunks,
)


def safe_print(msg: str) -> None:
    """Print message with UTF-8 encoding support for Urdu/Unicode text."""
    sys.stdout.buffer.write(msg.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_json_array(text: str) -> List[str]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    raw = match.group(0)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        return []
    return []


def extract_json_object(text: str, strict=False) -> Dict:
    """Extract JSON object from text with better error handling."""
    if not text or not isinstance(text, str):
        return {}

    # Try to find and extract JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    raw = match.group(0)

    # Try to parse directly
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # If strict parsing fails, try removing incomplete structures
    if not strict:
        # Try to fix common issues: remove trailing commas, incomplete arrays
        for attempt in range(3):
            try:
                fixed = re.sub(r",\s*([}\]])", r"\1", raw)  # Remove trailing commas
                data = json.loads(fixed)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                raw = raw[:-1]  # Remove last char and retry

    return {}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[۔.!؟?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def generate_fallback_claims_and_questions(
    answer: str,
    query: str,
    context_text: str,
) -> tuple:
    """Generate synthetic claims (by splitting answer) and alternate questions when LLM fails."""
    # Split answer into sentences/claims
    sentences = split_sentences(answer)
    claims = []
    verifications = []

    for sent in sentences[:4]:  # Max 4 claims
        if len(sent.strip()) < 10:
            continue

        # Simple heuristic: check if claim text appears in context
        verdict = (
            "SUPPORTED" if sent.lower() in context_text.lower() else "NOT_SUPPORTED"
        )

        # If not direct match, check token overlap
        if verdict == "NOT_SUPPORTED":
            sent_tokens = set(re.findall(r"\S+", sent.lower()))
            ctx_tokens = set(re.findall(r"\S+", context_text.lower()))
            overlap = (
                len(sent_tokens & ctx_tokens) / len(sent_tokens) if sent_tokens else 0
            )
            if overlap >= 0.4:
                verdict = "SUPPORTED"

        claims.append(sent.strip())
        verifications.append(
            {
                "claim": sent.strip(),
                "verdict": verdict,
                "reason": "Auto-generated fallback claim due to LLM JSON extraction failure.",
            }
        )

    # Generate alternate questions
    alt_questions = [
        f"{query} کی تفصیلات کیا ہیں؟",
        f"{query} سے بچاؤ کے طریقے کیا ہیں؟",
        f"{query} کے علاج یا انتظام کے بارے میں کیا معلومات ہیں؟",
    ]

    return verifications, alt_questions


def heuristic_evaluate_single_query(
    query: str,
    retrieval_args: argparse.Namespace,
    embedder: SentenceTransformer,
) -> Dict:
    hits = retrieve_chunks(retrieval_args, embedder=embedder)
    context_text = "\n\n".join([h.get("text", "") for h in hits if h.get("text")])

    # Build a lightweight fallback answer from top context when HF API is unavailable.
    top_text = hits[0].get("text", "") if hits else ""
    sentences = split_sentences(top_text)
    answer = (
        " ".join(sentences[:3]).strip()
        if sentences
        else "دی گئی معلومات میں واضح جواب موجود نہیں۔"
    )

    claim_candidates = split_sentences(answer)
    claims = claim_candidates[:3]
    verifications = []
    for claim in claims:
        verdict = "SUPPORTED" if claim and claim in context_text else "NOT_SUPPORTED"
        if verdict == "NOT_SUPPORTED":
            q_tokens = set(re.findall(r"\S+", claim))
            c_tokens = set(re.findall(r"\S+", context_text))
            overlap = (len(q_tokens & c_tokens) / len(q_tokens)) if q_tokens else 0.0
            if overlap >= 0.35:
                verdict = "SUPPORTED"
        verifications.append(
            {
                "claim": claim,
                "verdict": verdict,
                "reason": "Heuristic fallback verdict due to API unavailability.",
            }
        )

    supported = sum(1 for v in verifications if v["verdict"] == "SUPPORTED")
    faithfulness = (supported / len(verifications)) if verifications else 0.0

    alt_questions = [
        f"{query} کے بارے میں بنیادی معلومات کیا ہیں؟",
        f"{query} سے بچاؤ یا احتیاطی تدابیر کیا ہیں؟",
        f"{query} کی تشخیص یا علاج کے اہم نکات کیا ہیں؟",
    ]
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    rel_scores = []
    for q in alt_questions:
        q_emb = embedder.encode([q], normalize_embeddings=True)[0]
        rel_scores.append(cosine(query_emb, q_emb))
    relevancy = float(np.mean(rel_scores)) if rel_scores else 0.0

    return {
        "query": query,
        "used_generation_model": "heuristic-fallback",
        "used_judge_model": "heuristic-fallback",
        "answer": answer,
        "claims": claims,
        "claim_verification": verifications,
        "faithfulness_score": round(faithfulness, 4),
        "alternate_questions": alt_questions,
        "relevancy_similarities": [round(s, 4) for s in rel_scores],
        "relevancy_score": round(relevancy, 4),
        "retrieved_context": [
            {
                "chunk_id": h.get("chunk_id", ""),
                "title": h.get("title", ""),
                "source_file": h.get("source_file", ""),
                "text": h.get("text", ""),
            }
            for h in hits
        ],
        "fallback_mode": "heuristic",
    }


def llm_judge_once(
    query: str,
    answer: str,
    context_text: str,
    model_id: str,
    fallback_models: List[str],
    token: str,
) -> Dict:
    prompt = f"""
آپ ایک جج ہیں۔ نیچے سوال، جواب، اور سیاق دیا گیا ہے۔
آپ نے صرف JSON object واپس کرنا ہے (کوئی اضافی متن نہیں):

{{
  "claims": [
    {{"claim": "...", "verdict": "SUPPORTED یا NOT_SUPPORTED", "reason": "مختصر وجہ"}}
  ],
  "alternate_questions": ["...", "...", "..."]
}}

قواعد:
1) جواب سے 3 تا 6 قابلِ تصدیق دعوے نکالیں۔
2) ہر دعوے کو صرف دیے گئے سیاق کی بنیاد پر SUPPORTED یا NOT_SUPPORTED کریں۔
3) اسی جواب/سوال کی بنیاد پر 3 متبادل سوالات دیں۔

اصل سوال:
{query}

جواب:
{answer}

سیاق:
{context_text}
""".strip()

    text, _, used_model = call_github_models_with_fallback(
        prompt=prompt,
        primary_model=model_id,
        fallback_models=fallback_models,
        github_token=token,
        max_new_tokens=420,
        temperature=0.0,
        top_p=1.0,
    )

    obj = extract_json_object(text)
    claims_raw = obj.get("claims", []) if isinstance(obj, dict) else []
    alt_raw = obj.get("alternate_questions", []) if isinstance(obj, dict) else []

    verifications = []
    for item in claims_raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        verdict = str(item.get("verdict", "NOT_SUPPORTED")).strip().upper()
        reason = str(item.get("reason", "")).strip()
        if not claim:
            continue
        if verdict not in {"SUPPORTED", "NOT_SUPPORTED"}:
            verdict = "NOT_SUPPORTED"
        verifications.append({"claim": claim, "verdict": verdict, "reason": reason})

    alt_questions = [str(q).strip() for q in alt_raw if str(q).strip()][:3]

    # **FALLBACK: If JSON extraction failed, generate synthetic claims/questions**
    if not verifications or not alt_questions:
        safe_print(
            f"  [FALLBACK] JSON extraction failed or returned empty. Generating synthetic claims/questions."
        )
        verifications, alt_questions = generate_fallback_claims_and_questions(
            answer=answer,
            query=query,
            context_text=context_text,
        )

    return {
        "claim_verification": verifications,
        "alternate_questions": alt_questions,
        "used_judge_model": used_model,
    }


def evaluate_single_query(
    query: str,
    retrieval_args: argparse.Namespace,
    generation_model: str,
    generation_fallback_models: List[str],
    judge_model: str,
    judge_fallback_models: List[str],
    github_token: str,
    embedder: SentenceTransformer,
) -> Dict:
    hits = retrieve_chunks(retrieval_args, embedder=embedder)
    prompt = build_prompt(
        query, hits, max_context_chars=retrieval_args.max_context_chars
    )
    answer, _, used_generation_model = call_github_models_with_fallback(
        prompt=prompt,
        primary_model=generation_model,
        fallback_models=generation_fallback_models,
        github_token=github_token,
        max_new_tokens=retrieval_args.max_new_tokens,
        temperature=retrieval_args.temperature,
        top_p=retrieval_args.top_p,
    )

    context_text = "\n\n".join([h.get("text", "") for h in hits if h.get("text")])

    judged = llm_judge_once(
        query=query,
        answer=answer,
        context_text=context_text,
        model_id=judge_model,
        fallback_models=judge_fallback_models,
        token=github_token,
    )
    verifications = judged.get("claim_verification", [])
    claims = [v.get("claim", "") for v in verifications if v.get("claim")]

    supported = sum(1 for v in verifications if v["verdict"] == "SUPPORTED")
    faithfulness = (supported / len(verifications)) if verifications else 0.0

    alt_questions = judged.get("alternate_questions", [])
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    rel_scores = []
    for q in alt_questions:
        q_emb = embedder.encode([q], normalize_embeddings=True)[0]
        rel_scores.append(cosine(query_emb, q_emb))
    relevancy = float(np.mean(rel_scores)) if rel_scores else 0.0

    return {
        "query": query,
        "used_generation_model": used_generation_model,
        "used_judge_model": judged.get("used_judge_model", ""),
        "answer": answer,
        "claims": claims,
        "claim_verification": verifications,
        "faithfulness_score": round(faithfulness, 4),
        "alternate_questions": alt_questions,
        "relevancy_similarities": [round(s, 4) for s in rel_scores],
        "relevancy_score": round(relevancy, 4),
        "retrieved_context": [
            {
                "chunk_id": h.get("chunk_id", ""),
                "title": h.get("title", ""),
                "source_file": h.get("source_file", ""),
                "text": h.get("text", ""),
            }
            for h in hits
        ],
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate Urdu RAG with LLM-as-a-Judge (faithfulness + relevancy)."
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default="rag_artifacts/eval/test_queries_urdu.json",
        help="JSON file with list of test queries.",
    )
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
    parser.add_argument("--max-context-chars", type=int, default=10000)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--generation-model",
        type=str,
        default="openai/gpt-4.1-mini",
    )
    parser.add_argument(
        "--generation-fallback-models",
        type=str,
        default="",
        help="Comma-separated fallback generation models.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-4.1-mini",
        help="GitHub Models judge model for claim checks and alternate questions.",
    )
    parser.add_argument(
        "--judge-fallback-models",
        type=str,
        default="",
        help="Comma-separated fallback judge models.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=12,
        help="Maximum number of queries to evaluate from queries file.",
    )
    parser.add_argument(
        "--query-retries",
        type=int,
        default=2,
        help="How many retries per query on transient API failures.",
    )
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="If enabled, use heuristic fallback for queries that fail due API/provider issues.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="rag_artifacts/eval/eval_results.json",
    )
    args = parser.parse_args()

    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    if not github_token:
        raise RuntimeError("Missing GITHUB_TOKEN in environment or .env file.")

    q_path = Path(args.queries_file)
    if not q_path.exists():
        raise FileNotFoundError(f"Queries file not found: {q_path}")

    queries = read_json(q_path)
    if not isinstance(queries, list):
        raise RuntimeError("Queries file must contain a JSON list of question strings.")

    queries = [str(q).strip() for q in queries if str(q).strip()]
    queries = queries[: args.max_queries]
    if not queries:
        raise RuntimeError("No queries found to evaluate.")

    retrieval_args = argparse.Namespace(**vars(args))
    embedder = SentenceTransformer(args.embed_model)
    generation_fallback_models = parse_model_list(args.generation_fallback_models)
    judge_fallback_models = parse_model_list(args.judge_fallback_models)

    all_rows = []
    failed_rows = []
    heuristic_rows = 0
    for i, query in enumerate(queries, start=1):
        retrieval_args.query = query
        safe_print(f"Evaluating {i}/{len(queries)}: {query}")
        last_error = ""
        row = None
        for attempt in range(1, args.query_retries + 2):
            try:
                row = evaluate_single_query(
                    query=query,
                    retrieval_args=retrieval_args,
                    generation_model=args.generation_model,
                    generation_fallback_models=generation_fallback_models,
                    judge_model=args.judge_model,
                    judge_fallback_models=judge_fallback_models,
                    github_token=github_token,
                    embedder=embedder,
                )
                break
            except Exception as exc:
                last_error = str(exc)
                safe_print(f"  Attempt {attempt} failed: {last_error[:200]}")
                time.sleep(min(6, attempt * 2))

        if row is not None:
            all_rows.append(row)
        else:
            if args.allow_heuristic_fallback:
                row = heuristic_evaluate_single_query(
                    query=query,
                    retrieval_args=retrieval_args,
                    embedder=embedder,
                )
                all_rows.append(row)
                heuristic_rows += 1
                failed_rows.append(
                    {
                        "query": query,
                        "error": last_error[:500],
                        "resolved_with": "heuristic-fallback",
                    }
                )
            else:
                failed_rows.append({"query": query, "error": last_error[:500]})

    faith_scores = [r["faithfulness_score"] for r in all_rows] if all_rows else [0.0]
    rel_scores = [r["relevancy_score"] for r in all_rows] if all_rows else [0.0]
    summary = {
        "n_queries": len(queries),
        "n_success": len(all_rows),
        "n_failed": len(failed_rows),
        "n_heuristic_fallback": heuristic_rows,
        "avg_faithfulness": round(float(np.mean(faith_scores)), 4),
        "avg_relevancy": round(float(np.mean(rel_scores)), 4),
        "strategy": args.strategy,
        "bm25_only": args.bm25_only,
        "bm25_top_k": args.bm25_top_k,
        "semantic_top_k": args.semantic_top_k,
        "used_reranker": args.use_reranker,
        "generation_model": args.generation_model,
        "judge_model": args.judge_model,
    }

    payload = {
        "summary": summary,
        "results": all_rows,
        "failed": failed_rows,
    }

    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print with UTF-8 encoding
    import sys

    summary_str = json.dumps(summary, ensure_ascii=False, indent=2)
    sys.stdout.buffer.write(summary_str.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    print(f"Saved evaluation to {out}")


if __name__ == "__main__":
    main()
