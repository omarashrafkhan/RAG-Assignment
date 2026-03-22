import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p

    base = Path(__file__).resolve().parent
    c1 = base / p
    if c1.exists():
        return c1

    c2 = base.parent / p
    if c2.exists():
        return c2

    return p


def resolve_output_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    base = Path(__file__).resolve().parent
    c1 = base / p
    c2 = base.parent / p

    # Prefer current relative path when possible, else src-relative, else root-relative.
    if p.parent.exists() or p.parent == Path("."):
        return p
    if c1.parent.exists():
        return c1
    return c2


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

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    preferred_keys = {
        "claims",
        "claim_verification",
        "verifications",
        "claims_and_verdicts",
        "alternate_questions",
        "alternate_queries",
        "generated_questions",
    }

    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    first_obj: Optional[Dict] = None
    for start in starts:
        try:
            data, _ = decoder.raw_decode(text[start:])
            if isinstance(data, dict):
                if first_obj is None:
                    first_obj = data
                if any(k in data for k in preferred_keys):
                    return data
        except json.JSONDecodeError:
            continue

    if first_obj is not None:
        has_schema_hint = any(f'"{k}"' in text for k in preferred_keys)
        if has_schema_hint and not any(k in first_obj for k in preferred_keys):
            return {}
        return first_obj

    if strict:
        return {}

    first = text.find("{")
    if first == -1:
        return {}
    raw = text[first:]
    fixed = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        data, _ = decoder.raw_decode(fixed)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return {}
    return {}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[۔.!؟?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_claims_from_jsonish(text: str) -> List[Dict]:
    src = text or ""
    claim_pat = re.compile(r'"claim"\s*:\s*"(?P<claim>.*?)"', flags=re.DOTALL)
    verdict_pat = re.compile(r'"verdict"\s*:\s*"(?P<verdict>SUPPORTED|NOT_SUPPORTED)"')
    reason_pat = re.compile(r'"reason"\s*:\s*"(?P<reason>.*?)"', flags=re.DOTALL)

    rows: List[Dict] = []
    seen = set()
    claim_matches = list(claim_pat.finditer(src))

    for i, cm in enumerate(claim_matches):
        claim = re.sub(r"\s+", " ", cm.group("claim")).strip()
        if not claim:
            continue

        end = claim_matches[i + 1].start() if i + 1 < len(claim_matches) else len(src)
        segment = src[cm.end() : end]

        vm = verdict_pat.search(segment)
        verdict = normalize_verdict(vm.group("verdict") if vm else "NOT_SUPPORTED")

        rm = reason_pat.search(segment)
        reason = (
            re.sub(r"\s+", " ", rm.group("reason")).strip()
            if rm
            else "Recovered from truncated judge JSON output."
        )

        key = (claim, verdict)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"claim": claim, "verdict": verdict, "reason": reason})

    return rows


def parse_claim_lines(text: str) -> List[Dict]:
    rows: List[Dict] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().lstrip("-*")
        if not line:
            continue
        parts = re.split(r"\t|\|\|", line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            continue
        claim = parts[0]
        verdict = normalize_verdict(parts[1])
        reason = (
            parts[2] if len(parts) >= 3 else "Recovered from non-JSON judge output."
        )
        if claim:
            rows.append({"claim": claim, "verdict": verdict, "reason": reason})
    return rows


def normalize_verdict(raw_verdict: str) -> str:
    cleaned = re.sub(r"[^A-Z_]", "", str(raw_verdict).upper())
    if cleaned in {"SUPPORTED", "SUPPORT"}:
        return "SUPPORTED"
    if cleaned in {"NOTSUPPORTED", "NOT_SUPPORTED", "UNSUPPORTED"}:
        return "NOT_SUPPORTED"
    return "NOT_SUPPORTED"


def parse_judge_payload(obj: Dict) -> tuple:
    if not isinstance(obj, dict):
        return [], []

    # Some model outputs are a single claim object instead of {"claims": [...]}.
    if (
        "claims" not in obj
        and "claim_verification" not in obj
        and "verifications" not in obj
        and "claims_and_verdicts" not in obj
        and ("claim" in obj or "statement" in obj)
    ):
        obj = {"claims": [obj]}

    claims_raw = (
        obj.get("claims")
        or obj.get("claim_verification")
        or obj.get("verifications")
        or obj.get("claims_and_verdicts")
        or obj.get("دعوے")
        or []
    )
    alt_raw = (
        obj.get("alternate_questions")
        or obj.get("alternate_queries")
        or obj.get("generated_questions")
        or obj.get("متبادل_سوالات")
        or []
    )

    if isinstance(claims_raw, dict):
        claims_raw = [claims_raw]
    if isinstance(alt_raw, str):
        alt_raw = [alt_raw]

    verifications = []
    for item in claims_raw:
        if isinstance(item, dict):
            claim = str(
                item.get("claim", "")
                or item.get("text", "")
                or item.get("statement", "")
            ).strip()
            verdict = normalize_verdict(
                item.get("verdict", "NOT_SUPPORTED")
                or item.get("label", "NOT_SUPPORTED")
                or item.get("status", "NOT_SUPPORTED")
            )
            reason = str(item.get("reason", "") or item.get("why", "")).strip()
            if claim:
                verifications.append(
                    {"claim": claim, "verdict": verdict, "reason": reason}
                )
        elif isinstance(item, str) and item.strip():
            verifications.append(
                {
                    "claim": item.strip(),
                    "verdict": "NOT_SUPPORTED",
                    "reason": "Claim parsed from string-only judge output.",
                }
            )

    alt_questions = [str(q).strip() for q in alt_raw if str(q).strip()][:3]
    return verifications, alt_questions


def parse_claim_candidates(obj: Dict) -> List[str]:
    if not isinstance(obj, dict):
        return []

    raw = (
        obj.get("claims")
        or obj.get("claims_list")
        or obj.get("extracted_claims")
        or obj.get("statements")
        or []
    )

    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, dict):
        raw = [raw]

    claims: List[str] = []
    for item in raw:
        claim = ""
        if isinstance(item, dict):
            claim = str(
                item.get("claim", "")
                or item.get("text", "")
                or item.get("statement", "")
            ).strip()
        elif isinstance(item, str):
            claim = item.strip()
        if claim:
            claims.append(claim)

    deduped: List[str] = []
    seen = set()
    for c in claims:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped[:5]


def generate_fallback_claims_and_questions(
    answer: str,
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

    answer_sentences = split_sentences(answer)
    alt_questions = [
        f"{s[:90]} کے بارے میں مزید وضاحت کیا ہے؟" for s in answer_sentences[:3] if s
    ]
    if len(alt_questions) < 3:
        alt_questions.extend(
            [
                "دیے گئے جواب کے اہم نکات کیا ہیں؟",
                "جواب میں بیان کردہ معلومات کی سادہ تشریح کیا ہے؟",
                "جواب کے مطابق بنیادی طبی رہنمائی کیا بنتی ہے؟",
            ][: 3 - len(alt_questions)]
        )

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
    answer: str,
    context_text: str,
    model_id: str,
    fallback_models: List[str],
    token: str,
) -> Dict:
    used_models: List[str] = []
    sentence_count = len(split_sentences(answer))
    min_claims = 2 if sentence_count >= 2 else 1

    extract_claims_prompt = f"""
دیے گئے جواب سے اہم اور الگ الگ دعوے نکالیں۔

قواعد:
1) ہر دعویٰ ایک مکمل اور مختصر جملہ ہو۔
2) دعوے ایک دوسرے سے مختلف ہوں اور ایک ہی بات کو بار بار نہ دہرائیں۔
3) ہر دعویٰ جواب میں موجود معلومات پر مبنی ہو، کوئی نئی بات شامل نہ کریں۔
4) غیر ضروری یا کم اہم باتوں کو شامل نہ کریں۔
5) صرف valid JSON object دیں، کوئی اضافی متن نہ لکھیں۔

JSON:
{{
    "claims": ["دعویٰ 1", "دعویٰ 2", "..."]
}}

جواب:
{answer}
""".strip()

    claims_text, _, claims_model = call_github_models_with_fallback(
        prompt=extract_claims_prompt,
        primary_model=model_id,
        fallback_models=fallback_models,
        github_token=token,
        max_new_tokens=400,
        temperature=0.0,
        top_p=1.0,
        force_json=True,
    )
    if claims_model:
        used_models.append(claims_model)

    claims_obj = extract_json_object(claims_text)
    claim_candidates = parse_claim_candidates(claims_obj)
    if len(claim_candidates) < min_claims:
        claim_candidates = [s for s in split_sentences(answer)[:3] if s]

    claims_bulleted = "\n".join([f"- {c}" for c in claim_candidates])
    verify_prompt = f"""
نیچے دیے گئے دعوؤں کو صرف سیاق کی بنیاد پر چیک کریں۔

قواعد:
1) دعووں کا متن نہ بدلیں۔
2) ہر دعویٰ کے لیے verdict دیں: SUPPORTED یا NOT_SUPPORTED
3) reason مختصر رکھیں (زیادہ سے زیادہ 20 الفاظ)
4) صرف valid JSON object دیں۔

JSON:
{{
  "claims": [
    {{"claim": "...", "verdict": "SUPPORTED یا NOT_SUPPORTED", "reason": "..."}}
  ]
}}

دعوے:
{claims_bulleted}

سیاق:
{context_text}
""".strip()

    verify_text, _, verify_model = call_github_models_with_fallback(
        prompt=verify_prompt,
        primary_model=model_id,
        fallback_models=fallback_models,
        github_token=token,
        max_new_tokens=1000,
        temperature=0.0,
        top_p=1.0,
        force_json=True,
    )
    if verify_model:
        used_models.append(verify_model)

    verify_obj = extract_json_object(verify_text)
    verifications, _ = parse_judge_payload(verify_obj)
    if len(verifications) < min_claims:
        recovered_verify = parse_claims_from_jsonish(verify_text)
        if len(recovered_verify) > len(verifications):
            verifications = recovered_verify

        alt_prompt = f"""
آپ کا کام صرف متبادل سوالات بنانا ہے، اور وہ صرف جواب کی عبارت پر مبنی ہوں۔

اہم ہدایت:
1) جواب سے 3 متبادل سوالات بنائیں۔
2) صرف valid JSON object دیں۔

{{
    "alternate_questions": ["سوال 1", "سوال 2", "سوال 3"]
}}

جواب:
{answer}
""".strip()

    alt_text, _, alt_model = call_github_models_with_fallback(
        prompt=alt_prompt,
        primary_model=model_id,
        fallback_models=fallback_models,
        github_token=token,
        max_new_tokens=250,
        temperature=0.0,
        top_p=1.0,
        force_json=True,
    )
    if alt_model:
        used_models.append(alt_model)

    alt_obj = extract_json_object(alt_text)
    _, alt_questions = parse_judge_payload(alt_obj)

    # Single-pass mode: one call for alternate questions.

    # **FALLBACK: If JSON extraction failed, generate synthetic claims/questions**
    if len(verifications) < min_claims or not alt_questions:
        safe_print(
            "  [FALLBACK] JSON extraction failed or returned empty. Generating synthetic claims/questions."
        )
        verifications, alt_questions = generate_fallback_claims_and_questions(
            answer=answer,
            context_text=context_text,
        )

    return {
        "claim_verification": verifications,
        "alternate_questions": alt_questions,
        "used_judge_model": " + ".join(
            [m for i, m in enumerate(used_models) if m and m not in used_models[:i]]
        ),
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

    q_path = resolve_path(args.queries_file)
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

    out = resolve_output_path(args.save_json)
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
