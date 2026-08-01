import streamlit as st
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Document Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint Base URL
API_URL = "http://127.0.0.1:8000/api/v1"

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222a;
        border: 1px solid #2e3440;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: #00d2ff;
    }
    .metric-label {
        font-size: 12px;
        color: #a0a5b5;
        margin-bottom: 4px;
    }
    .sub-metric-card {
        background-color: #161920;
        border: 1px solid #282c34;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
    }
    .sub-metric-value {
        font-size: 15px;
        font-weight: bold;
        color: #70a1ff;
    }
    .sub-metric-label {
        font-size: 11px;
        color: #8b92a5;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Document Intelligence Engine")
st.caption("Hybrid RAG with Re-ranking, Latency Tracing & Token Cost Analysis")


# --- HELPER FUNCTIONS ---
def fetch_indexed_documents():
    """Fetch list of active indexed files from backend API."""
    try:
        res = requests.get(f"{API_URL}/documents")
        if res.status_code == 200:
            return res.json()
        return []
    except Exception:
        return []


def delete_document(doc_id: str):
    """Delete document from vector store via API."""
    try:
        res = requests.delete(f"{API_URL}/documents/{doc_id}")
        return res.status_code == 200
    except Exception:
        return False


def get_confidence_badge(score: float) -> tuple[str, str]:
    """Returns a status badge emoji and CSS color based on confidence thresholds."""
    if score >= 75.0:
        return "🟢 High", "#2ea043"      # GitHub Green
    elif score >= 50.0:
        return "🟡 Medium", "#d29922"    # GitHub Amber
    else:
        return "🔴 Low", "#cf222e"       # GitHub Red


def render_citations(citations: list[dict]):
    """Renders sources and citations with confidence badges."""
    if not citations:
        st.info("No sources retrieved for this query.")
        return

    with st.expander("📚 Sources & Citations", expanded=True):
        for idx, cite in enumerate(citations, start=1):
            file_name = cite.get("file_name", "Document")
            page_num = cite.get("page_number")
            row_num = cite.get("row_number")
            
            if row_num is not None:
                location_str = f"Row {row_num}"
            elif page_num is not None:
                location_str = f"Page {page_num}"
            else:
                location_str = "N/A"
            
            # Confidence score is normalized to 0-100 float
            score = cite.get("relevance_score", 0.0)
            badge_text, badge_color = get_confidence_badge(score)
            
            # Formatted Header
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.markdown(f"**{idx}. 📄 {file_name}** `({location_str})`")
            with header_col2:
                st.markdown(
                    f"<span style='color: {badge_color}; font-weight: bold; float: right;'>"
                    f"{badge_text} ({score:.1f}%)</span>",
                    unsafe_allow_html=True
                )

            # Display Text Snippet
            st.caption(f'"{cite.get("text_snippet", "").strip()}"')
            if idx < len(citations):
                st.divider()


# --- SIDEBAR: PIPELINE CONTROLS, UPLOAD & INDEXED FILES ---
with st.sidebar:
    st.header("🎛️ Pipeline Settings")
    
    st.subheader("Retrieval Controls")
    top_k = st.slider("Top-K Final Chunks", min_value=1, max_value=10, value=4)
    
    st.divider()
    
    st.subheader("📂 Document Management")
    uploaded_file = st.file_uploader(
        "Upload document (.pdf, .txt, .md, .csv)",
        type=["pdf", "txt", "md", "csv"]
    )
    
    if uploaded_file is not None:
        if st.button("Index Document", type="primary", use_container_width=True):
            with st.spinner("Processing & indexing vectors..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    res = requests.post(f"{API_URL}/documents/upload", files=files)

                    if res.status_code == 200:
                        st.success(f"Successfully indexed: {uploaded_file.name}")
                        st.rerun()  # Refresh sidebar list
                    else:
                        st.error(f"Upload failed ({res.status_code}): {res.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Could not connect to backend API: {e}")

    st.divider()

    # --- INDEXED DOCUMENTS MANAGER ---
    st.subheader("📚 Active Indexed Documents")
    
    indexed_docs = fetch_indexed_documents()
    
    if not indexed_docs:
        st.info("No documents currently indexed in ChromaDB.")
    else:
        st.caption(f"Total Files Indexed: **{len(indexed_docs)}**")
        for doc in indexed_docs:
            d_id = doc.get("doc_id", "")
            f_name = doc.get("file_name", "Unknown File")
            chunks = doc.get("chunk_count", 0)
            pages = doc.get("total_pages", 1)

            with st.container(border=True):
                st.markdown(f"**📄 {f_name}**")
                st.caption(f"Chunks: `{chunks}` | Pages/Units: `{pages}`")
                
                if st.button("🗑️ Delete File", key=f"del_{d_id}", use_container_width=True):
                    if delete_document(d_id):
                        st.success(f"Deleted {f_name}")
                        st.rerun()
                    else:
                        st.error("Failed to delete document.")

# --- MAIN LAYOUT: QUERY & TELEMETRY ---
query_input = st.text_area(
    "Ask a question about your indexed documents:",
    placeholder="What are the key findings or performance metrics in the report?",
    height=100
)

col_run, _ = st.columns([1, 5])
with col_run:
    submit_btn = st.button("🚀 Execute Query", type="primary")

if submit_btn and query_input.strip():
    with st.spinner("Running Dense Search + BM25 + Cross-Encoder Rerank + LLM..."):
        try:
            payload = {
                "prompt": query_input,
                "top_k": top_k
            }
            
            response = requests.post(f"{API_URL}/query", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                citations = data.get("citations", [])
                metrics = data.get("metrics", {})
                retrieval_breakdown = metrics.get("retrieval_breakdown", {})

                # --- 1. RESPONSE DISPLAY ---
                st.subheader("🤖 Generated Answer")
                st.markdown(f"> {answer}")

                st.divider()

                # --- 2. TELEMETRY & OBSERVABILITY DASHBOARD ---
                st.subheader("📊 Execution Telemetry & Latency Breakdown")
                
                m1, m2, m3, m4, m5 = st.columns(5)
                
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Retrieval Time</div>
                        <div class="metric-value">{metrics.get('retrieval_latency_ms', 0):.1f} ms</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">LLM Latency</div>
                        <div class="metric-value">{metrics.get('llm_latency_ms', 0):.1f} ms</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Latency</div>
                        <div class="metric-value">{metrics.get('total_latency_ms', 0):.1f} ms</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m4:
                    prompt_toks = metrics.get('prompt_tokens', 0)
                    compl_toks = metrics.get('completion_tokens', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Tokens</div>
                        <div class="metric-value">{prompt_toks + compl_toks}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m5:
                    cost = metrics.get('estimated_cost_usd', 0.0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Est. Cost</div>
                        <div class="metric-value">${cost:.6f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Granular Retrieval Latency Deep Dive
                if retrieval_breakdown:
                    with st.expander("🔍 Detailed Retrieval Stage Latency Breakdown", expanded=False):
                        r1, r2, r3, r4, r5 = st.columns(5)
                        with r1:
                            st.markdown(f"""
                            <div class="sub-metric-card">
                                <div class="sub-metric-label">Dense Encode</div>
                                <div class="sub-metric-value">{retrieval_breakdown.get('dense_ms', 0.0):.1f} ms</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with r2:
                            st.markdown(f"""
                            <div class="sub-metric-card">
                                <div class="sub-metric-label">ChromaDB I/O</div>
                                <div class="sub-metric-value">{retrieval_breakdown.get('chroma_io_ms', 0.0):.1f} ms</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with r3:
                            st.markdown(f"""
                            <div class="sub-metric-card">
                                <div class="sub-metric-label">BM25 Lookup</div>
                                <div class="sub-metric-value">{retrieval_breakdown.get('bm25_ms', 0.0):.1f} ms</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with r4:
                            st.markdown(f"""
                            <div class="sub-metric-card">
                                <div class="sub-metric-label">RRF Fusion</div>
                                <div class="sub-metric-value">{retrieval_breakdown.get('fusion_ms', 0.0):.1f} ms</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with r5:
                            st.markdown(f"""
                            <div class="sub-metric-card">
                                <div class="sub-metric-label">Reranking</div>
                                <div class="sub-metric-value">{retrieval_breakdown.get('rerank_ms', 0.0):.1f} ms</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Visual Latency Progress Bar
                total_t = max(metrics.get('total_latency_ms', 1.0), 1.0)
                ret_pct = int((metrics.get('retrieval_latency_ms', 0) / total_t) * 100)
                llm_pct = int((metrics.get('llm_latency_ms', 0) / total_t) * 100)
                
                st.caption(f"**Pipeline Execution Split:** Retrieval Phase ~{ret_pct}% | LLM Generation Phase ~{llm_pct}%")

                st.divider()

                # --- 3. CITATIONS & RE-RANKER SCORES ---
                render_citations(citations)

            else:
                st.error(f"Error {response.status_code}: {response.json().get('detail', 'Backend server error')}")

        except Exception as e:
            st.error(f"Failed to query backend server. Make sure FastAPI server is running on port 8000. Error: {e}")