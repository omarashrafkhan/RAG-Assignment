import streamlit as st
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional
import numpy as np
import json
import re
import time
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_answer import (
    retrieve_chunks,
    build_prompt,
    call_github_models_with_fallback,
)
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

GENERATION_MODEL = "openai/gpt-4.1-mini"
GENERATION_FALLBACK_MODELS = ["openai/gpt-4.1-nano"]
JUDGE_MODEL = "openai/gpt-4.1-mini"
JUDGE_FALLBACK_MODELS = ["openai/gpt-4.1-nano"]


# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Urdu Medical RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional appearance
st.markdown(
    """
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: #2c3e50;
        margin-top: 1.5rem;
    }
    
    /* Cards/Boxes */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    .context-box {
        background: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 0.25rem;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.25rem;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
    }
    
    .stButton > button:hover {
        background-color: #0d47a1;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# Initialize Session & Config
# ==============================================================================
load_dotenv()


@st.cache_resource
def load_models():
    """Load embedder model once."""
    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return embedder


@st.cache_resource
def get_retrieval_args():
    """Create retrieval arguments."""
    args = argparse.Namespace(
        query="",
        strategy="sentence",
        chunk_file="",
        index_name="urdu-medical-rag",
        namespace="",
        embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        bm25_top_k=30,
        semantic_top_k=30,
        rrf_k=60,
        final_top_k=5,
        bm25_only=False,
        use_reranker=True,
        reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        max_context_chars=10000,
        max_new_tokens=200,
        temperature=0.2,
        top_p=0.9,
    )
    return args


@st.cache_data
def load_test_queries() -> List[str]:
    base = Path(__file__).resolve().parent
    path = base / "rag_artifacts" / "eval" / "test_queries_urdu.json"
    if not path.exists():
        # HF Spaces often runs from repo root while app is under src/
        path = base.parent / "rag_artifacts" / "eval" / "test_queries_urdu.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(q).strip() for q in data if str(q).strip()]


@st.cache_data
def load_ablation_rows() -> List[Dict[str, str]]:
    base = Path(__file__).resolve().parent
    path = base / "rag_artifacts" / "eval" / "ablation_table.csv"
    if not path.exists():
        path = base.parent / "rag_artifacts" / "eval" / "ablation_table.csv"
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: str(v) for k, v in row.items()})
    return rows


def find_ablation_row(strategy: str, retrieval_mode: str) -> Optional[Dict[str, str]]:
    rows = load_ablation_rows()
    if not rows:
        return None

    retrieval_label = (
        "Semantic-only"
        if retrieval_mode == "Semantic-Only"
        else "Hybrid (BM25+Semantic+RRF)"
    )
    reranking_label = "No" if retrieval_mode == "Semantic-Only" else "Yes"

    for row in rows:
        if (
            row.get("strategy", "") == strategy
            and row.get("retrieval_mode", "") == retrieval_label
            and row.get("reranking", "") == reranking_label
        ):
            return row
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def extract_json_object(text: str) -> Dict:
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
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                if first_obj is None:
                    first_obj = obj
                if any(k in obj for k in preferred_keys):
                    return obj
        except Exception:
            continue

    if first_obj is not None:
        has_schema_hint = any(f'"{k}"' in text for k in preferred_keys)
        if has_schema_hint and not any(k in first_obj for k in preferred_keys):
            # Top-level schema likely truncated; avoid locking onto nested claim objects.
            return {}
        return first_obj

    # Last attempt: trim common issues and retry from first brace.
    first = text.find("{")
    if first == -1:
        return {}
    raw = text[first:]
    fixed = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        obj, _ = decoder.raw_decode(fixed)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[۔.!؟?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def looks_like_truncated_json(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    if s.count("{") > s.count("}") or s.count("[") > s.count("]"):
        return True
    if len(re.findall(r'(?<!\\)"', s)) % 2 != 0:
        return True
    return s[-1] not in {"}", "]", '"'}


def token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\S+", (a or "").lower()))
    b_tokens = set(re.findall(r"\S+", (b or "").lower()))
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def parse_claim_lines(text: str) -> List[Dict]:
    verifications: List[Dict] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().lstrip("-*")
        if not line:
            continue
        # Expected shape: claim<TAB or ||>verdict<TAB or ||>reason
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
            verifications.append(
                {
                    "claim": claim,
                    "verdict": verdict,
                    "reason": reason,
                }
            )
    return verifications


def parse_claims_from_jsonish(text: str) -> List[Dict]:
    # Recover claims even when trailing objects are truncated.
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

    alt_questions = [str(q).strip() for q in alt_raw if str(q).strip()]
    alt_questions = alt_questions[:3]

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


def extract_citation_ids(answer: str) -> List[int]:
    ids = re.findall(r"\[(\d+)\]", answer or "")
    uniq = sorted({int(x) for x in ids if x.isdigit()})
    return uniq


def judge_answer_with_llm(
    query: str,
    answer: str,
    context_text: str,
    github_token: str,
    embedder: SentenceTransformer,
) -> Dict:
    used_models: List[str] = []
    debug_info: Dict = {
        "faithfulness": {},
        "alternate_questions": {},
    }

    sentence_count = len(split_sentences(answer))
    min_claims = 2 if sentence_count >= 2 else 1
    debug_info["faithfulness"]["min_claims_required"] = min_claims

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
        primary_model=JUDGE_MODEL,
        fallback_models=JUDGE_FALLBACK_MODELS,
        github_token=github_token,
        max_new_tokens=400,
        temperature=0.0,
        top_p=1.0,
        force_json=True,
    )
    if claims_model:
        used_models.append(claims_model)
    debug_info["faithfulness"]["claims_model"] = claims_model
    debug_info["faithfulness"]["claims_raw_length"] = len(claims_text or "")
    debug_info["faithfulness"]["claims_raw_preview"] = (claims_text or "")[:2000]
    debug_info["faithfulness"][
        "claims_response_complete"
    ] = not looks_like_truncated_json(claims_text)

    claims_obj = extract_json_object(claims_text)
    claim_candidates = parse_claim_candidates(claims_obj)
    if len(claim_candidates) < min_claims:
        claim_candidates = [s for s in split_sentences(answer)[:3] if s]
        debug_info["faithfulness"]["claims_heuristic_used"] = True
    debug_info["faithfulness"]["claims_json_keys"] = (
        list(claims_obj.keys()) if isinstance(claims_obj, dict) else []
    )
    debug_info["faithfulness"]["claims_count"] = len(claim_candidates)

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
        primary_model=JUDGE_MODEL,
        fallback_models=JUDGE_FALLBACK_MODELS,
        github_token=github_token,
        max_new_tokens=1000,
        temperature=0.0,
        top_p=1.0,
        force_json=True,
    )
    if verify_model:
        used_models.append(verify_model)
    debug_info["faithfulness"]["verify_model"] = verify_model
    debug_info["faithfulness"]["verify_raw_length"] = len(verify_text or "")
    debug_info["faithfulness"]["verify_raw_preview"] = (verify_text or "")[:2000]
    debug_info["faithfulness"][
        "verify_response_complete"
    ] = not looks_like_truncated_json(verify_text)

    verify_obj = extract_json_object(verify_text)
    verifications, _ = parse_judge_payload(verify_obj)
    if len(verifications) < min_claims:
        recovered_verify = parse_claims_from_jsonish(verify_text)
        if len(recovered_verify) > len(verifications):
            verifications = recovered_verify
    debug_info["faithfulness"]["verify_json_keys"] = (
        list(verify_obj.keys()) if isinstance(verify_obj, dict) else []
    )
    debug_info["faithfulness"]["verify_claim_count"] = len(verifications)

    if len(verifications) < min_claims:
        fallback_claims = (
            claim_candidates if claim_candidates else split_sentences(answer)[:3]
        )
        verifications = []
        for c in fallback_claims:
            if not c:
                continue
            overlap = token_overlap_ratio(c, context_text)
            verdict = (
                "SUPPORTED"
                if (c in context_text or overlap >= 0.35)
                else "NOT_SUPPORTED"
            )
            verifications.append(
                {
                    "claim": c,
                    "verdict": verdict,
                    "reason": f"Heuristic fallback: verify parse failed; overlap={overlap:.2f}",
                }
            )
        debug_info["faithfulness"]["heuristic_used"] = True

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
        primary_model=JUDGE_MODEL,
        fallback_models=JUDGE_FALLBACK_MODELS,
        github_token=github_token,
        max_new_tokens=250,
        temperature=0.0,
        top_p=1.0,
        force_json=True,
    )
    if alt_model:
        used_models.append(alt_model)
    debug_info["alternate_questions"]["first_model"] = alt_model
    debug_info["alternate_questions"]["first_raw_length"] = len(alt_text or "")
    debug_info["alternate_questions"]["first_raw_preview"] = (alt_text or "")[:1000]
    debug_info["alternate_questions"]["first_response_complete"] = not (
        alt_text or ""
    ).rstrip().endswith("[") and not (alt_text or "").rstrip().endswith(",")

    alt_obj = extract_json_object(alt_text)
    _, alt_questions = parse_judge_payload(alt_obj)
    debug_info["alternate_questions"]["first_json_keys"] = (
        list(alt_obj.keys()) if isinstance(alt_obj, dict) else []
    )
    debug_info["alternate_questions"]["first_count"] = len(alt_questions)

    debug_info["pipeline_mode"] = "single_pass_4_calls"

    if not alt_questions:
        answer_sentences = split_sentences(answer)
        alt_questions = [
            f"{s[:90]} کے بارے میں مزید وضاحت کیا ہے؟"
            for s in answer_sentences[:3]
            if s
        ]
        if len(alt_questions) < 3:
            alt_questions.extend(
                [
                    "دیے گئے جواب کے اہم نکات کیا ہیں؟",
                    "جواب میں بیان کردہ معلومات کی سادہ تشریح کیا ہے؟",
                    "جواب کے مطابق بنیادی طبی رہنمائی کیا بنتی ہے؟",
                ][: 3 - len(alt_questions)]
            )
        debug_info["alternate_questions"]["heuristic_used"] = True

    supported = sum(1 for v in verifications if v.get("verdict") == "SUPPORTED")
    faithfulness = (supported / len(verifications)) if verifications else 0.0

    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    rel_scores = []
    for alt_q in alt_questions:
        q_emb = embedder.encode([alt_q], normalize_embeddings=True)[0]
        rel_scores.append(cosine_similarity(query_emb, q_emb))
    relevancy = float(np.mean(rel_scores)) if rel_scores else 0.0

    unique_models = [
        m for i, m in enumerate(used_models) if m and m not in used_models[:i]
    ]

    return {
        "faithfulness": round(faithfulness, 4),
        "relevancy": round(relevancy, 4),
        "claim_verification": verifications,
        "alternate_questions": alt_questions,
        "relevancy_similarities": [round(s, 4) for s in rel_scores],
        "used_judge_model": " + ".join(unique_models),
        "debug": debug_info,
    }


# ==============================================================================
# Main UI Layout
# ==============================================================================
st.title("🏥 Urdu Medical RAG System")
st.markdown("*Retrieval-Augmented Generation for Medical Question Answering*")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    strategy = st.selectbox(
        "Chunking Strategy",
        options=["fixed", "recursive", "sentence"],
        help="Document chunking approach",
    )
    retrieval_mode = st.radio(
        "Retrieval Mode",
        options=["Semantic-Only", "Hybrid+Re-ranking"],
        index=1,
        help="Choose retrieval pipeline behavior",
    )

    st.markdown("---")
    st.subheader("Demo Query Presets")
    presets = load_test_queries()
    if presets:
        selected_preset = st.selectbox("Choose a test query", options=presets)
        if st.button("Use Preset", use_container_width=True):
            st.session_state["query_input"] = selected_preset
    else:
        st.caption("No preset queries found.")

summary_row = find_ablation_row(strategy=strategy, retrieval_mode=retrieval_mode)
st.markdown("### Experiment Summary")
sum_c1, sum_c2, sum_c3, sum_c4, sum_c5, sum_c6 = st.columns(6)
sum_c1.metric("Chunking", strategy)
sum_c2.metric("Retrieval", retrieval_mode)
sum_c3.metric("Generation", GENERATION_MODEL)
sum_c4.metric("Judge", JUDGE_MODEL)
if summary_row:
    try:
        sum_c5.metric(
            "Expected Faith", f"{float(summary_row.get('avg_faithfulness', '0')):.2%}"
        )
    except Exception:
        sum_c5.metric("Expected Faith", "N/A")
    try:
        sum_c6.metric(
            "Expected Relevancy", f"{float(summary_row.get('avg_relevancy', '0')):.2%}"
        )
    except Exception:
        sum_c6.metric("Expected Relevancy", "N/A")
else:
    sum_c5.metric("Expected Faith", "N/A")
    sum_c6.metric("Expected Relevancy", "N/A")

# Main query section
st.markdown("### 🔍 Query Input")
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_area(
        "Enter your medical question in Urdu:",
        placeholder="مثال: ذیابیطس کی علامات کیا ہیں؟",
        height=80,
        label_visibility="collapsed",
        key="query_input",
    )

with col2:
    submit_btn = st.button("Submit", use_container_width=True, type="primary")

# Process query
if submit_btn and query.strip():
    with st.spinner("🔄 Processing your query..."):
        try:
            latency: Dict[str, float] = {}
            pipeline_logs: List[str] = []
            total_start = time.perf_counter()

            # Load models
            embedder = load_models()
            retrieval_args = get_retrieval_args()
            retrieval_args.query = query
            retrieval_args.strategy = strategy
            pipeline_logs.append(
                f"Query length={len(query)} | strategy={strategy} | mode={retrieval_mode}"
            )

            if retrieval_mode == "Semantic-Only":
                retrieval_args.bm25_only = False
                retrieval_args.bm25_top_k = 0
                retrieval_args.semantic_top_k = 30
                retrieval_args.use_reranker = False
            else:
                retrieval_args.bm25_only = False
                retrieval_args.bm25_top_k = 30
                retrieval_args.semantic_top_k = 30
                retrieval_args.use_reranker = True

            with st.status("Running pipeline...", expanded=True) as status:
                status.write("Stage 1/4: Retrieving chunks (BM25/Semantic/RRF)...")
                retrieval_start = time.perf_counter()
                hits = retrieve_chunks(retrieval_args, embedder=embedder)
                latency["retrieval"] = round(time.perf_counter() - retrieval_start, 3)
                pipeline_logs.append(
                    f"Retrieved {len(hits)} chunks in {latency['retrieval']:.3f}s"
                )

                context_text = "\n\n".join(
                    [h.get("text", "") for h in hits if h.get("text")]
                )
                pipeline_logs.append(f"Context chars={len(context_text)}")

                status.write("Stage 2/4: Building prompt and generating answer...")
                prompt = build_prompt(query, hits, max_context_chars=10000)
                pipeline_logs.append(f"Generation prompt chars={len(prompt)}")

                github_token = os.getenv("GITHUB_TOKEN", "").strip()
                generation_start = time.perf_counter()
                answer, _, model_used = call_github_models_with_fallback(
                    prompt=prompt,
                    primary_model=GENERATION_MODEL,
                    fallback_models=GENERATION_FALLBACK_MODELS,
                    github_token=github_token,
                    max_new_tokens=200,
                    temperature=0.2,
                    top_p=0.9,
                )
                latency["generation"] = round(time.perf_counter() - generation_start, 3)
                pipeline_logs.append(
                    f"Generation model={model_used} | answer chars={len(answer)} | generation {latency['generation']:.3f}s"
                )

                status.write("Stage 3/4: Running LLM-as-Judge...")
                judge_start = time.perf_counter()
                metrics = judge_answer_with_llm(
                    query=query,
                    answer=answer,
                    context_text=context_text,
                    github_token=github_token,
                    embedder=embedder,
                )
                latency["judge"] = round(time.perf_counter() - judge_start, 3)
                pipeline_logs.append(
                    f"Judge models={metrics.get('used_judge_model', 'N/A')} | judge {latency['judge']:.3f}s"
                )
                pipeline_logs.append(
                    f"Claims={len(metrics.get('claim_verification', []))} | AltQuestions={len(metrics.get('alternate_questions', []))}"
                )

                latency["total"] = round(time.perf_counter() - total_start, 3)
                status.write("Stage 4/4: Finalizing results...")
                status.update(label="Pipeline complete", state="complete")

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                [
                    "📄 Answer",
                    "🔗 Context",
                    "🧠 Explainability",
                    "⚖️ Judge Details",
                    "📊 Metrics",
                ]
            )

            with tab1:
                st.markdown("#### Generated Answer")
                st.markdown(
                    f"""
                <div class="answer-box">
                    {answer}
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Generated using: {model_used}")

                citation_ids = extract_citation_ids(answer)
                with st.expander("Citations Map"):
                    if not citation_ids:
                        st.caption("No inline citations found in the answer.")
                    for cid in citation_ids:
                        idx = cid - 1
                        if 0 <= idx < len(hits):
                            h = hits[idx]
                            st.markdown(
                                f"**[{cid}]** {h.get('title', 'Unknown')} | {h.get('source_file', 'N/A')} | {h.get('chunk_id', 'N/A')}"
                            )
                        else:
                            st.markdown(
                                f"**[{cid}]** No matching retrieved chunk in current top-{len(hits)}."
                            )

            with tab2:
                st.markdown(f"#### Retrieved Context ({len(hits)} chunks)")
                for i, hit in enumerate(hits, 1):
                    with st.expander(f"📌 Chunk {i} - {hit.get('title', 'Unknown')}"):
                        st.markdown(f"**Source:** {hit.get('source_file', 'N/A')}")
                        st.markdown(f"**Chunk ID:** `{hit.get('chunk_id', 'N/A')}`")
                        st.markdown("---")
                        st.markdown(
                            f"""
                        <div class="context-box">
                            {hit.get("text", "")[:500]}...
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

            with tab3:
                st.markdown("#### Retrieval Explainability")
                explain_rows = []
                for i, h in enumerate(hits, 1):
                    explain_rows.append(
                        {
                            "citation": i,
                            "chunk_id": h.get("chunk_id", ""),
                            "source_file": h.get("source_file", ""),
                            "bm25_rank": h.get("bm25_rank", None),
                            "semantic_rank": h.get("semantic_rank", None),
                            "rrf": h.get("rrf", None),
                            "rerank_score": h.get("rerank_score", None),
                        }
                    )
                st.dataframe(explain_rows, use_container_width=True)

            with tab4:
                st.markdown("#### Claim Verification")
                for i, v in enumerate(metrics.get("claim_verification", []), 1):
                    st.markdown(
                        f"**{i}. {v.get('verdict', 'NOT_SUPPORTED')}** - {v.get('claim', '')}"
                    )
                    reason = v.get("reason", "")
                    if reason:
                        st.caption(reason)

                st.markdown("---")
                st.markdown("#### Alternate Questions")
                alts = metrics.get("alternate_questions", [])
                sims = metrics.get("relevancy_similarities", [])
                for i, q in enumerate(alts):
                    sim = sims[i] if i < len(sims) else None
                    if sim is None:
                        st.markdown(f"{i + 1}. {q}")
                    else:
                        st.markdown(f"{i + 1}. {q} (sim={sim:.4f})")
                st.caption(f"Judge model: {metrics.get('used_judge_model', 'N/A')}")
                with st.expander("Judge Debug Payload"):
                    st.json(metrics.get("debug", {}))

            with tab5:
                st.markdown("#### Performance Metrics")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                    <div class="metric-box">
                        <h3>Faithfulness</h3>
                        <h1>{metrics["faithfulness"]:.2%}</h1>
                        <p>Claim verification score</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                    <div class="metric-box">
                        <h3>Relevancy</h3>
                        <h1>{metrics["relevancy"]:.2%}</h1>
                        <p>Answer relevance to query</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Additional info
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Retrieved Chunks", len(hits))
                col2.metric("Mode", retrieval_mode)
                col3.metric(
                    "Re-ranking", "✓" if retrieval_mode == "Hybrid+Re-ranking" else "✗"
                )

                st.markdown("---")
                st.markdown("#### Latency Telemetry")
                lt1, lt2, lt3, lt4 = st.columns(4)
                lt1.metric("Retrieval (s)", f"{latency.get('retrieval', 0.0):.3f}")
                lt2.metric("Generation (s)", f"{latency.get('generation', 0.0):.3f}")
                lt3.metric("Judge (s)", f"{latency.get('judge', 0.0):.3f}")
                lt4.metric("Total (s)", f"{latency.get('total', 0.0):.3f}")

                with st.expander("Pipeline Debug Logs"):
                    st.code("\n".join(pipeline_logs) if pipeline_logs else "No logs")

        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.info(
                "Please check your GITHUB_TOKEN environment variable and try again."
            )

elif submit_btn:
    st.warning("⚠️ Please enter a query in Urdu first.")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Urdu Medical RAG System</strong> | Built with Streamlit, Pinecone & LLMs</p>
    <p>For questions or issues, refer to the project documentation.</p>
</div>
""",
    unsafe_allow_html=True,
)
