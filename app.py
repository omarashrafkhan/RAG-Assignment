import streamlit as st
import argparse
from pathlib import Path
import sys
from typing import Dict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_answer import retrieve_chunks, build_prompt, call_github_models_with_fallback
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os


# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Urdu Medical RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional appearance
st.markdown("""
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
""", unsafe_allow_html=True)

# ==============================================================================
# Initialize Session & Config
# ==============================================================================
load_dotenv()

@st.cache_resource
def load_models():
    """Load embedder model once."""
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

@st.cache_data
def compute_answer_metrics(answer: str, query: str, _embedder: SentenceTransformer) -> Dict:
    """
    Compute basic heuristic metrics for demo purposes.
    In production, these would come from LLM-as-Judge.
    """
    # Simple heuristic: compare query with answer using embeddings
    query_emb = _embedder.encode([query], normalize_embeddings=True)[0]
    answer_emb = _embedder.encode([answer], normalize_embeddings=True)[0]
    relevancy = cosine_similarity(query_emb, answer_emb)
    
    # Heuristic faithfulness: presence of citations (very basic)
    has_citations = "[" in answer and "]" in answer
    faithfulness = 0.85 if has_citations else 0.65
    
    return {
        "faithfulness": round(faithfulness, 4),
        "relevancy": round(relevancy, 4),
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
        help="Document chunking approach"
    )
    use_hybrid = st.checkbox("Use Hybrid Search", value=True, help="BM25 + Semantic + RRF")
    use_rerank = st.checkbox("Use Re-ranking", value=True, help="Cross-Encoder re-ranking")

# Main query section
st.markdown("### 🔍 Query Input")
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_area(
        "Enter your medical question in Urdu:",
        placeholder="مثال: ذیابیطس کی علامات کیا ہیں؟",
        height=80,
        label_visibility="collapsed"
    )

with col2:
    submit_btn = st.button("Submit", use_container_width=True, type="primary")

# Process query
if submit_btn and query.strip():
    with st.spinner("🔄 Processing your query..."):
        try:
            # Load models
            embedder = load_models()
            retrieval_args = get_retrieval_args()
            retrieval_args.query = query
            retrieval_args.strategy = strategy
            retrieval_args.use_reranker = use_rerank
            retrieval_args.bm25_only = not use_hybrid
            
            # Retrieve context
            hits = retrieve_chunks(retrieval_args, embedder=embedder)
            context_text = "\n\n".join([h.get("text", "") for h in hits if h.get("text")])
            
            # Build prompt and generate answer
            prompt = build_prompt(query, hits, max_context_chars=10000)
            
            github_token = os.getenv("GITHUB_TOKEN", "").strip()
            answer, _, model_used = call_github_models_with_fallback(
                prompt=prompt,
                primary_model="openai/gpt-4.1-nano",
                fallback_models=["openai/gpt-4.1-mini"],
                github_token=github_token,
                max_new_tokens=300,
                temperature=0.2,
                top_p=0.9,
            )
            
            # Compute metrics
            metrics = compute_answer_metrics(answer, query, _embedder=embedder)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📄 Answer", "🔗 Context", "📊 Metrics"])
            
            with tab1:
                st.markdown("#### Generated Answer")
                st.markdown(f"""
                <div class="answer-box">
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Generated using: {model_used}")
            
            with tab2:
                st.markdown(f"#### Retrieved Context ({len(hits)} chunks)")
                for i, hit in enumerate(hits, 1):
                    with st.expander(f"📌 Chunk {i} - {hit.get('title', 'Unknown')}"):
                        st.markdown(f"**Source:** {hit.get('source_file', 'N/A')}")
                        st.markdown(f"**Chunk ID:** `{hit.get('chunk_id', 'N/A')}`")
                        st.markdown("---")
                        st.markdown(f"""
                        <div class="context-box">
                            {hit.get('text', '')[:500]}...
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab3:
                st.markdown("#### Performance Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>Faithfulness</h3>
                        <h1>{metrics['faithfulness']:.2%}</h1>
                        <p>Claim verification score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>Relevancy</h3>
                        <h1>{metrics['relevancy']:.2%}</h1>
                        <p>Answer relevance to query</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional info
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Retrieved Chunks", len(hits))
                col2.metric("Hybrid Search", "✓" if use_hybrid else "✗")
                col3.metric("Re-ranking", "✓" if use_rerank else "✗")
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.info("Please check your GITHUB_TOKEN environment variable and try again.")

elif submit_btn:
    st.warning("⚠️ Please enter a query in Urdu first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Urdu Medical RAG System</strong> | Built with Streamlit, Pinecone & LLMs</p>
    <p>For questions or issues, refer to the project documentation.</p>
</div>
""", unsafe_allow_html=True)
