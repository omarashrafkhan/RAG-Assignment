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
    initial_sidebar_state="collapsed",
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
        max_new_tokens=300,
        temperature=0.2,
        top_p=0.9,
    )
    return args


@st.cache_data
def load_test_queries() -> List[str]:
    path = Path("rag_artifacts/eval/test_queries_urdu.json")
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
    path = Path("rag_artifacts/eval/ablation_table.csv")
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

    # Remove optional fenced-code wrappers if the model returns markdown.
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    raw = match.group(0)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        # Try light repair for common JSON issues from LLM outputs.
        for _ in range(3):
            try:
                fixed = re.sub(r",\s*([}\]])", r"\1", raw)
                obj = json.loads(fixed)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                raw = raw[:-1]
        return {}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[۔.!؟?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


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
    prompt = f"""
آپ ایک جج ہیں جو دعوے کی سچائی کو دیے گئے سیاق و سباق کی بنیاد پر پرکھتا ہے۔

نیچے سوال، جواب، اور سیاق دیا گیا ہے۔

آپ نے صرف اور صرف ایک درست JSON object واپس کرنا ہے۔ کوئی اضافی متن نہ دیں۔

{{
  "claims": [
        {{"claim": "یہ دعویٰ یہاں", "verdict": "SUPPORTED یا NOT_SUPPORTED", "reason": "مختصر وجہ یہاں"}}
  ],
    "alternate_questions": ["متبادل سوال 1", "متبادل سوال 2"]
}}

قواعد:
1) جواب سے اہم اور قابل تصدیق دعوے نکالیں۔ تعداد پر خود فیصلہ کریں۔
2) ہر دعوے کے لیے صرف دیے گئے سیاق کی بنیاد پر SUPPORTED یا NOT_SUPPORTED دیں۔
3) وجہ کو مختصر اور واضح لکھیں، صرف حقائق پر مبنی۔
4) اسی جواب/سوال کی بنیاد پر متبادل سوالات دیں، تعداد خود منتخب کریں۔
5) JSON کو ہمیشہ درست اور parse-able رکھیں۔

اصل سوال:
{query}

جواب:
{answer}

سیاق:
{context_text}
""".strip()

    text, _, used_judge_model = call_github_models_with_fallback(
        prompt=prompt,
        primary_model=JUDGE_MODEL,
        fallback_models=JUDGE_FALLBACK_MODELS,
        github_token=github_token,
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

    # Robust fallback to avoid empty metrics when judge output is malformed.
    if not verifications:
        fallback_claims = split_sentences(answer)[:3]
        verifications = [
            {
                "claim": c,
                "verdict": "SUPPORTED" if c and c in context_text else "NOT_SUPPORTED",
                "reason": "Fallback verification due to malformed judge output.",
            }
            for c in fallback_claims
            if c
        ]
    if not alt_questions:
        alt_questions = [
            f"{query} کی بنیادی وضاحت کیا ہے؟",
            f"{query} کے اسباب یا خطرات کیا ہیں؟",
            f"{query} سے بچاؤ یا علاج کے اہم نکات کیا ہیں؟",
        ]

    supported = sum(1 for v in verifications if v.get("verdict") == "SUPPORTED")
    faithfulness = (supported / len(verifications)) if verifications else 0.0

    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    rel_scores = []
    for q in alt_questions:
        q_emb = embedder.encode([q], normalize_embeddings=True)[0]
        rel_scores.append(cosine_similarity(query_emb, q_emb))
    relevancy = float(np.mean(rel_scores)) if rel_scores else 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "relevancy": round(relevancy, 4),
        "claim_verification": verifications,
        "alternate_questions": alt_questions,
        "relevancy_similarities": [round(s, 4) for s in rel_scores],
        "used_judge_model": used_judge_model,
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
            total_start = time.perf_counter()

            # Load models
            embedder = load_models()
            retrieval_args = get_retrieval_args()
            retrieval_args.query = query
            retrieval_args.strategy = strategy

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

                context_text = "\n\n".join(
                    [h.get("text", "") for h in hits if h.get("text")]
                )

                status.write("Stage 2/4: Building prompt and generating answer...")
                prompt = build_prompt(query, hits, max_context_chars=10000)

                github_token = os.getenv("GITHUB_TOKEN", "").strip()
                generation_start = time.perf_counter()
                answer, _, model_used = call_github_models_with_fallback(
                    prompt=prompt,
                    primary_model=GENERATION_MODEL,
                    fallback_models=GENERATION_FALLBACK_MODELS,
                    github_token=github_token,
                    max_new_tokens=300,
                    temperature=0.2,
                    top_p=0.9,
                )
                latency["generation"] = round(time.perf_counter() - generation_start, 3)

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
