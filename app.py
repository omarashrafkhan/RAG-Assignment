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
import os

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_answer import (
    retrieve_chunks,
    build_prompt,
    call_github_models_with_fallback,
)
from sentence_transformers import SentenceTransformer

GENERATION_MODEL = "openai/gpt-4.1-mini"
GENERATION_FALLBACK_MODELS = ["openai/gpt-4.1-nano"]
JUDGE_MODEL = "openai/gpt-4.1-mini"
JUDGE_FALLBACK_MODELS = ["openai/gpt-4.1-nano"]


# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Urdu Medical RAG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# Custom Theme State Management (Top Right Toggle)
# ==============================================================================
if "is_dark_mode" not in st.session_state:
    st.session_state.is_dark_mode = True

col_spacer, col_toggle = st.columns([10, 2])
with col_toggle:
    st.session_state.is_dark_mode = st.toggle(
        "🌙 Dark Mode", value=st.session_state.is_dark_mode
    )

if st.session_state.is_dark_mode:
    THEME_CSS = """
    :root {
        --bg-main: #171717;
        --bg-surface: #212121;
        --bg-elevated: #2f2f2f;
        --border: #424242;
        --text-primary: #ececec;
        --text-secondary: #a3a3a3;
        --accent: #10a37f;
        --accent-glow: rgba(16, 163, 127, 0.2);
    }
    """
else:
    THEME_CSS = """
    :root {
        --bg-main: #f9f9fb;
        --bg-surface: #ffffff;
        --bg-elevated: #f0f0f4;
        --border: #e5e7eb;
        --text-primary: #111827;
        --text-secondary: #4b5563;
        --accent: #10a37f;
        --accent-glow: rgba(16, 163, 127, 0.1);
    }
    """

st.markdown(
    f"""
<style>
    /* ── Import Fonts ───────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    /* ── Theme Definitions ─────────────────────────────────────────── */
    {THEME_CSS}

    /* ── Hide Streamlit Chrome Completely ──────────────────────────── */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden; height: 0;}}
    footer {{visibility: hidden;}}
    section[data-testid="stSidebar"] {{display: none;}}

    /* ── Global Structure & Width Restrictions ─────────────────────── */
    .stApp, .main {{
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }}

    .block-container {{
        padding: 0 1rem 1rem 1rem !important;
        max-width: 98% !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }}

    p, span, li, label, .stMarkdown {{
        color: var(--text-primary) !important;
    }}

    .material-icons {{
        font-family: 'Material Icons';
        font-weight: normal;
        font-style: normal;
        display: inline-block;
        line-height: 1;
        text-transform: none;
        letter-spacing: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
        vertical-align: middle;
        margin-right: 0.3rem; 
        color: inherit;
    }}

    /* ── Data Elements ─────────────────────────────────────────────── */
    div[data-testid="stMetricLabel"] {{
        white-space: nowrap !important;
        overflow: visible !important;
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--text-primary) !important;
        font-size: 1.4rem !important;
    }}

    /* ── Inputs and Controls ───────────────────────────────────────── */
    .stTextArea textarea {{
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0.75rem !important;
        font-family: 'Noto Nastaliq Urdu', serif !important;
        font-size: 1.15rem !important;
        line-height: 1.8 !important;
        padding: 1rem !important;
        direction: rtl !important;
        text-align: right !important;
        height: 80px !important;
        box-shadow: 0 2px 4px var(--accent-glow) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .stTextArea textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }}

    /* Selectbox Main Box Context */
    div[data-baseweb="select"] > div, 
    div[role="radiogroup"] label {{
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
    }}

    /* Selectbox Dropdown Menu Context (Light/Dark Mode Fix) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div {{
        background-color: var(--bg-surface) !important;
        border-radius: 4px;
    }}
    
    ul[role="listbox"],
    ul[data-baseweb="menu"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    li[role="option"] {{
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
    }}

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    li[aria-selected="true"] {{
        background-color: var(--bg-elevated) !important;
        color: var(--accent) !important;
    }}

    /* ── Main Aesthetics: Center Header & Logo ────────────────────── */
    .center-header {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-top: -1rem; 
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
    }}

    .icon-wrapper {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 80px;
        height: 80px;
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: 0 8px 16px var(--accent-glow);
        margin-bottom: 1rem;
    }}

    .main-title {{
        font-size: 2.5rem !important;
        margin-bottom: 0 !important;
        background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .urdu-logo {{
        font-family: 'Noto Nastaliq Urdu', serif !important;
        font-size: 1.6rem !important;
        color: var(--accent) !important;
        margin-top: 0.5rem !important;
        font-weight: 700 !important;
    }}

    /* ── Fixed Scrollable Output Area ──────────────────────────────── */
    .assistant-message {{
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        padding: 1.5rem 2rem;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.1rem;
        line-height: 2.2;
        color: var(--text-primary);
        max-height: 40vh;
        overflow-y: auto;
        box-shadow: 0 4px 6px -1px var(--accent-glow);
    }}

    .assistant-message::-webkit-scrollbar {{
        width: 6px;
    }}
    .assistant-message::-webkit-scrollbar-track {{
        background: var(--bg-main);
    }}
    .assistant-message::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 4px;
    }}
    .assistant-message::-webkit-scrollbar-thumb:hover {{
        background: var(--accent);
    }}

    .chunk-text {{
        direction: rtl; 
        text-align: right; 
        font-family: 'Noto Nastaliq Urdu', serif; 
        max-height: 250px; 
        overflow-y: auto;
        color: var(--text-primary);
    }}

    /* ── Badges ───────────────────────────────────────────────────── */
    .verdict-tag {{
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        margin-left: 0.5rem;
        display: inline-block;
    }}
    .verdict-tag.supported {{ background: rgba(16, 163, 127, 0.15); color: #10a37f; }}
    .verdict-tag.not-supported {{ background: rgba(239, 68, 68, 0.15); color: #ef4444; }}

    .streamlit-expanderContent {{
        background-color: var(--bg-main) !important;
        border-color: var(--border) !important;
    }}

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return embedder

@st.cache_resource
def get_retrieval_args():
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
    retrieval_label = "Semantic-only" if retrieval_mode == "Semantic-Only" else "Hybrid (BM25+Semantic+RRF)"
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
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

def extract_json_object(text: str) -> Dict:
    if not text: return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    preferred_keys = {"claims", "claim_verification", "verifications", "claims_and_verdicts", "alternate_questions", "alternate_queries", "generated_questions"}
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    first_obj: Optional[Dict] = None
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                if first_obj is None: first_obj = obj
                if any(k in obj for k in preferred_keys): return obj
        except: pass
    if first_obj is not None: return first_obj
    first = text.find("{")
    if first == -1: return {}
    raw = text[first:]
    fixed = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        obj, _ = decoder.raw_decode(fixed)
        return obj if isinstance(obj, dict) else {}
    except: return {}

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
        if not claim: continue
        end = claim_matches[i + 1].start() if i + 1 < len(claim_matches) else len(src)
        segment = src[cm.end() : end]
        vm = verdict_pat.search(segment)
        verdict = "SUPPORTED" if vm and vm.group("verdict") == "SUPPORTED" else "NOT_SUPPORTED"
        rm = reason_pat.search(segment)
        reason = re.sub(r"\s+", " ", rm.group("reason")).strip() if rm else "Recovered."
        key = (claim, verdict)
        if key in seen: continue
        seen.add(key)
        rows.append({"claim": claim, "verdict": verdict, "reason": reason})
    return rows

def judge_answer_with_llm(query: str, answer: str, context_text: str, github_token: str, embedder: SentenceTransformer) -> Dict:
    used_models: List[str] = []
    sentence_count = len(split_sentences(answer))
    min_claims = 2 if sentence_count >= 2 else 1
    
    extract_claims_prompt = f"""
دیے گئے جواب سے اہم اور الگ الگ دعوے نکالیں۔
JSON: {{"claims": ["دعویٰ 1", "دعویٰ 2", "..."]}}
جواب: {answer}
""".strip()
    claims_text, _, claims_model = call_github_models_with_fallback(
        prompt=extract_claims_prompt, primary_model=JUDGE_MODEL, fallback_models=JUDGE_FALLBACK_MODELS,
        github_token=github_token, max_new_tokens=400, temperature=0.0, top_p=1.0, force_json=True,
    )
    if claims_model: used_models.append(claims_model)
    claims_obj = extract_json_object(claims_text)
    
    raw = claims_obj.get("claims", [])
    if isinstance(raw, str): raw = [raw]
    claim_candidates = [str(c) for c in raw if isinstance(c, str) and c.strip()][:5]
    if len(claim_candidates) < min_claims: claim_candidates = [s for s in split_sentences(answer)[:3] if s]
    
    claims_bulleted = "\n".join([f"- {c}" for c in claim_candidates])
    verify_prompt = f"""
نیچے دیے گئے دعوؤں کو صرف سیاق کی بنیاد پر چیک کریں۔
JSON: {{"claims": [{{"claim": "...", "verdict": "SUPPORTED یا NOT_SUPPORTED", "reason": "..."}}]}}
دعوے:\n{claims_bulleted}\n\nسیاق:\n{context_text}
""".strip()
    verify_text, _, verify_model = call_github_models_with_fallback(
        prompt=verify_prompt, primary_model=JUDGE_MODEL, fallback_models=JUDGE_FALLBACK_MODELS,
        github_token=github_token, max_new_tokens=1000, temperature=0.0, top_p=1.0, force_json=True,
    )
    if verify_model: used_models.append(verify_model)
    
    verify_obj = extract_json_object(verify_text)
    raw_v = verify_obj.get("claims", [])
    if isinstance(raw_v, dict): raw_v = [raw_v]
    verifications = []
    for item in raw_v:
        if isinstance(item, dict):
            c = item.get("claim", "")
            v = "SUPPORTED" if "SUPPORT" in str(item.get("verdict", "")).upper() else "NOT_SUPPORTED"
            r = item.get("reason", "")
            if c: verifications.append({"claim": c, "verdict": v, "reason": r})
            
    if len(verifications) < min_claims:
        recovered = parse_claims_from_jsonish(verify_text)
        if len(recovered) > len(verifications): verifications = recovered
        
    alt_prompt = f"""
جواب سے 3 متبادل سوالات بنائیں۔
{{ "alternate_questions": ["سوال 1", "سوال 2", "سوال 3"] }}
جواب: {answer}
""".strip()
    alt_text, _, alt_model = call_github_models_with_fallback(
        prompt=alt_prompt, primary_model=JUDGE_MODEL, fallback_models=JUDGE_FALLBACK_MODELS,
        github_token=github_token, max_new_tokens=250, temperature=0.0, top_p=1.0, force_json=True,
    )
    if alt_model: used_models.append(alt_model)
    alt_obj = extract_json_object(alt_text)
    alt_questions = [str(q) for q in alt_obj.get("alternate_questions", []) if str(q).strip()][:3]
    
    supported = sum(1 for v in verifications if v.get("verdict") == "SUPPORTED")
    faithfulness = (supported / len(verifications)) if verifications else 0.0
    
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    rel_scores = []
    for alt_q in alt_questions:
        q_emb = embedder.encode([alt_q], normalize_embeddings=True)[0]
        rel_scores.append(cosine_similarity(query_emb, q_emb))
    relevancy = float(np.mean(rel_scores)) if rel_scores else 0.0
    
    unique_models = [m for i, m in enumerate(used_models) if m and m not in used_models[:i]]
    return {
        "faithfulness": round(faithfulness, 4),
        "relevancy": round(relevancy, 4),
        "claim_verification": verifications,
        "alternate_questions": alt_questions,
        "relevancy_similarities": [round(s, 4) for s in rel_scores],
        "used_judge_model": " + ".join(unique_models),
        "debug": {},
    }

def extract_citation_ids(answer: str) -> List[int]:
    ids = re.findall(r"\[(\d+)\]", answer or "")
    return sorted({int(x) for x in ids if x.isdigit()})


# ==============================================================================
# Refined UI Structure
# ==============================================================================

# ── Aesthetic Central Logo ──
st.markdown(
    """
<div class="center-header">
    <div class="icon-wrapper">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="var(--accent)">
            <path d="M16 4h-2V2h-4v2H8C6.9 4 6 4.9 6 6v14c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm-1 10h-2v2h-2v-2H9v-2h2v-2h2v2h2v2z"/>
        </svg>
    </div>
    <h1 class="main-title">Urdu Medical RAG</h1>
    <h2 class="urdu-logo">ذیابیطس میڈیکل اسسٹنٹ</h2>
</div>
""",
    unsafe_allow_html=True,
)


# ── Configuration & Presets ──
st.markdown("### <span class='material-icons'>settings</span> Configuration & Presets", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    strategy = st.selectbox("Chunking Strategy", ["fixed", "recursive", "sentence"], index=2, label_visibility="visible")
with c2:
    retrieval_mode = st.radio("Retrieval Mode", ["Semantic-Only", "Hybrid+Re-ranking"], index=1, horizontal=True)
with c3:
    presets = load_test_queries()
    selected_preset = st.selectbox("Load Test Query", [""] + presets) if presets else ""
    if selected_preset: st.session_state["query_input"] = selected_preset

st.markdown("---")

# ── Experiment Summary ──
st.markdown("### <span class='material-icons'>analytics</span> Experiment Summary", unsafe_allow_html=True)
summary_row = find_ablation_row(strategy=strategy, retrieval_mode=retrieval_mode)

col_s1, col_s2, col_s3, col_s4, col_s5, col_s6 = st.columns(6, gap="small")
col_s1.metric("Chunking", strategy)
col_s2.metric("Retrieval", "Hybrid+Rerank" if "Hybrid" in retrieval_mode else "Semantic")
col_s3.metric("Generation", GENERATION_MODEL.split('/')[-1])
col_s4.metric("Judge", JUDGE_MODEL.split('/')[-1])
col_s5.metric("Exp. Faith", f"{float(summary_row.get('avg_faithfulness', '0')):.2%}" if summary_row else "--")
col_s6.metric("Exp. Relevancy", f"{float(summary_row.get('avg_relevancy', '0')):.2%}" if summary_row else "--")

st.markdown("---")

# ── Query Input ──
col_input, col_btn = st.columns([5, 1])
with col_input:
    st.markdown('<div class="urdu-input">', unsafe_allow_html=True)
    query = st.text_area(
        "Query",
        placeholder="اپنا طبی سوال یہاں لکھیں...",
        label_visibility="collapsed",
        key="query_input",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_btn:
    st.write("")
    st.write("")
    submit_btn = st.button("Submit", type="primary", use_container_width=True)

# ==============================================================================
# Processing Pipeline 
# ==============================================================================
if submit_btn and query.strip():
    with st.spinner("Processing medical corpus..."):
        try:
            latency: Dict[str, float] = {}
            total_start = time.perf_counter()

            embedder = load_models()
            retrieval_args = get_retrieval_args()
            retrieval_args.query = query
            retrieval_args.strategy = strategy
            retrieval_args.bm25_only = False
            retrieval_args.bm25_top_k = 0 if retrieval_mode == "Semantic-Only" else 30
            retrieval_args.semantic_top_k = 30
            retrieval_args.use_reranker = retrieval_mode != "Semantic-Only"

            with st.status("Thinking...", expanded=False) as status:
                status.write("Retrieving documents...")
                ret_s = time.perf_counter()
                hits = retrieve_chunks(retrieval_args, embedder=embedder)
                latency["retrieval"] = round(time.perf_counter() - ret_s, 3)
                context_text = "\n\n".join([h.get("text", "") for h in hits if h.get("text")])
                
                status.write("Generating answer...")
                prompt = build_prompt(query, hits, max_context_chars=10000)
                github_token = os.getenv("GITHUB_TOKEN", "").strip()
                gen_s = time.perf_counter()
                answer, _, model_used = call_github_models_with_fallback(
                    prompt=prompt, primary_model=GENERATION_MODEL, fallback_models=GENERATION_FALLBACK_MODELS,
                    github_token=github_token, max_new_tokens=200, temperature=0.2, top_p=0.9,
                )
                latency["generation"] = round(time.perf_counter() - gen_s, 3)
                
                status.write("Judging quality...")
                judge_s = time.perf_counter()
                metrics = judge_answer_with_llm(
                    query=query, answer=answer, context_text=context_text,
                    github_token=github_token, embedder=embedder,
                )
                latency["judge"] = round(time.perf_counter() - judge_s, 3)
                latency["total"] = round(time.perf_counter() - total_start, 3)
                status.update(label=f"Done in {latency['total']}s", state="complete")

            # Store results in session state so they persist across Theme toggles (Script reruns)
            st.session_state["rag_results"] = {
                "hits": hits,
                "answer": answer,
                "model_used": model_used,
                "metrics": metrics,
                "latency": latency,
            }

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

elif submit_btn:
    st.info("Please enter a question in Urdu to proceed.")


# ==============================================================================
# Render Results from Session State (Persists across toggle reruns)
# ==============================================================================
if "rag_results" in st.session_state:
    res = st.session_state["rag_results"]
    hits = res["hits"]
    answer = res["answer"]
    model_used = res["model_used"]
    metrics = res["metrics"]
    latency = res["latency"]

    tab_ans, tab_ctx, tab_exp, tab_judge, tab_met = st.tabs([
        "Answer", "Context", "Explainability", "Judge Details", "Metrics"
    ])

    with tab_ans:
        st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)
        cids = extract_citation_ids(answer)
        if cids:
            st.write("**Citations Map**")
            for cid in cids:
                idx = cid - 1
                if 0 <= idx < len(hits): st.write(f"**[{cid}]** {hits[idx].get('source_file', 'N/A')}")

    with tab_ctx:
        st.markdown(f"**{len(hits)} chunks retrieved**")
        for i, hit in enumerate(hits, 1):
            with st.expander(f"[{i}] {hit.get('source_file', 'Source')}"):
                st.markdown(f"<div class='chunk-text'>{hit.get('text', '')}</div>", unsafe_allow_html=True)

    with tab_exp:
        explain_rows = [{
            "Cite": i, "Chunk ID": h.get("chunk_id", ""), "Source": h.get("source_file", ""),
            "Semantic Rank": h.get("semantic_rank", None),
            "RRF": round(h.get("rrf", 0), 4) if "rrf" in h else None,
            "Rerank Score": round(h.get("rerank_score", 0), 4) if "rerank_score" in h else None,
        } for i, h in enumerate(hits, 1)]
        st.dataframe(explain_rows, use_container_width=True, height=250)

    with tab_judge:
        for v in metrics.get("claim_verification", []):
            verdict, c, r = v.get("verdict", "NOT_SUPPORTED"), v.get('claim', ''), v.get('reason', '')
            cls = "supported" if verdict == "SUPPORTED" else "not-supported"
            icn = "verified" if verdict == "SUPPORTED" else "error"
            st.markdown(
                f"""
                <div style="direction: rtl; text-align: right; margin-bottom: 0.5rem; font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.1rem; color: var(--text-primary);">
                    <span class="verdict-tag {cls}"><span class="material-icons" style="font-size: 0.8rem; vertical-align:middle;">{icn}</span> {verdict}</span> 
                    {c}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem; border-left: 2px solid var(--border); padding-left: 0.5rem;">Reasoning: {r}</div>
                """, unsafe_allow_html=True
            )

    with tab_met:
        m1, m2 = st.columns(2)
        m1.metric("Faithfulness Score", f"{metrics['faithfulness']:.0%}")
        m2.metric("Relevancy Score", f"{metrics['relevancy']:.0%}")
        st.markdown("---")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Retrieval Latency", f"{latency.get('retrieval', 0):.2f}s")
        l2.metric("Generation Latency", f"{latency.get('generation', 0):.2f}s")
        l3.metric("Judge Latency", f"{latency.get('judge', 0):.2f}s")
        l4.metric("Total Latency", f"{latency.get('total', 0):.2f}s")
