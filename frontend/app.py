import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backend.pipeline import MediRAGPipeline

st.set_page_config(
    page_title="MediRAG | Clinical Healthcare Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Theme CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #0b4f6c;
        padding-bottom: 5px;
    }
    .sub-header {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .emergency-box {
        background-color: #feebe8;
        border-left: 5px solid #d32f2f;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 15px;
        color: #b71c1c;
        font-weight: 500;
    }
    .citation-badge {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
        border: 1px solid #bae6fd;
    }
    .citation-badge-verified {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .evidence-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0fdfa;
        border: 1px solid #ccfbf1;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return MediRAGPipeline.get_instance()

pipeline = get_pipeline()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None

# Sidebar Configuration
with st.sidebar:
    st.title("🩺 MediRAG")
    st.caption("Evidence-Grounded Clinical Information Assistant")
    st.divider()

    st.subheader("⚙️ Retrieval Settings")
    mode_options = {
        "Hybrid + Reranker (Recommended)": "hybrid_rerank",
        "Hybrid (Vector + BM25 RRF)": "hybrid",
        "Dense Vector (FAISS)": "vector",
        "Sparse Keyword (BM25)": "bm25"
    }
    selected_mode_label = st.selectbox(
        "Retrieval Architecture:",
        options=list(mode_options.keys()),
        index=0
    )
    search_mode = mode_options[selected_mode_label]

    top_k = st.slider("Evidence Passages (Top-K):", min_value=1, max_value=8, value=4)

    st.divider()
    st.subheader("📊 System Status")
    stats = pipeline.get_stats()
    st.write(f"**Indexed Documents:** `{stats['total_documents']}`")
    st.write(f"**Total Evidence Chunks:** `{stats['total_chunks']}`")
    st.write(f"**Total Pages:** `{stats['total_pages']}`")
    st.write(f"**LLM Model:** `{stats['llm_model']}`")
    st.write(f"**Embedding Model:** `all-MiniLM-L6-v2`")
    st.write(f"**Reranker:** `ms-marco-MiniLM-L-6-v2`")

    st.divider()
    if st.button("🗑️ Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_query_result = None
        st.rerun()

# Main Tabs
tab_chat, tab_docs, tab_evidence, tab_eval = st.tabs([
    "💬 Clinical Assistant",
    "📚 Document Management",
    "🔍 Evidence Explorer",
    "📈 Evaluation Dashboard"
])

# ----------------- TAB 1: CHAT INTERFACE -----------------
with tab_chat:
    st.markdown("<h2 class='main-header'>MediRAG Healthcare Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Ask questions about hypertension, diabetes, asthma, cardiovascular conditions, or uploaded clinical documents.</div>", unsafe_allow_html=True)

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.markdown("**Source Citations:**")
                citation_html = ""
                for c in msg["citations"]:
                    badge_class = "citation-badge citation-badge-verified" if c.get("is_grounded") else "citation-badge"
                    status = "✓ Grounded" if c.get("is_grounded") else "Cited"
                    citation_html += f"<span class='{badge_class}'>📄 {c.get('doc_name')} (p.{c.get('page_number')}) - {status}</span>"
                st.markdown(citation_html, unsafe_allow_html=True)

    # Prompt Input
    user_query = st.chat_input("Enter a medical or health-related query...")

    # Follow-up buttons trigger
    if st.session_state.last_query_result and st.session_state.last_query_result.get("follow_up_questions"):
        st.write("💡 **Suggested Follow-up Inquiries:**")
        cols = st.columns(len(st.session_state.last_query_result["follow_up_questions"]))
        for idx, fq in enumerate(st.session_state.last_query_result["follow_up_questions"]):
            if cols[idx].button(fq, key=f"fq_{idx}"):
                user_query = fq

    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant thinking & retrieval
        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving evidence using {selected_mode_label}..."):
                # Pass conversation history for context
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                result = pipeline.query(
                    question=user_query,
                    search_mode=search_mode,
                    top_k=top_k,
                    history=history
                )
                st.session_state.last_query_result = result

            # Render output
            st.markdown(result["answer"])

            # Render citation badges
            if result.get("citations"):
                st.markdown("**Source Citations:**")
                badges = ""
                for c in result["citations"]:
                    is_gr = c.get("is_grounded", False)
                    cls_name = "citation-badge citation-badge-verified" if is_gr else "citation-badge"
                    tag = "✓ Verified Source" if is_gr else "Referenced"
                    badges += f"<span class='{cls_name}'>📄 {c.get('doc_name')} (p.{c.get('page_number')}) • {tag}</span>"
                st.markdown(badges, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result.get("citations", [])
        })
        st.rerun()

# ----------------- TAB 2: DOCUMENT MANAGEMENT -----------------
with tab_docs:
    st.subheader("📚 Knowledge Base & Document Repository")
    st.write("Upload medical documents (PDF, DOCX, TXT) to automatically parse, chunk, embed, and index into the MediRAG pipeline.")

    # Upload section
    uploaded_file = st.file_uploader("Upload New Clinical Document:", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        if st.button("📥 Process & Index Document", type="primary"):
            save_path = settings.RAW_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Extracting, chunking, and embedding {uploaded_file.name}..."):
                try:
                    record = pipeline.ingest_file(save_path)
                    st.success(f"Successfully indexed '{record.filename}'! Generated {record.num_chunks} chunks across {record.num_pages} pages.")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to index document: {e}")

    st.divider()
    st.subheader("Indexed Documents")
    docs = pipeline.get_documents()
    if not docs:
        st.info("No documents currently indexed.")
    else:
        doc_data = []
        for d in docs:
            doc_data.append({
                "Document Name": d.filename,
                "Format": d.file_type.upper(),
                "Pages": d.num_pages,
                "Chunks": d.num_chunks,
                "Size (KB)": round(d.file_size / 1024, 1),
                "Uploaded At": d.uploaded_at[:19].replace("T", " ")
            })
        st.dataframe(pd.DataFrame(doc_data), use_container_width=True)

        st.write("### Manage Document Removal")
        del_col1, del_col2 = st.columns([3, 1])
        selected_to_delete = del_col1.selectbox("Select Document to Remove:", options=[d.doc_id for d in docs], format_func=lambda x: f"{x} ({pipeline.document_store.get_document(x).filename if pipeline.document_store.get_document(x) else x})")
        if del_col2.button("🗑️ Delete Document", type="secondary"):
            with st.spinner(f"Removing {selected_to_delete} and recalculating indices..."):
                pipeline.delete_file(selected_to_delete)
                st.success(f"Document '{selected_to_delete}' removed.")
                time.sleep(1)
                st.rerun()

# ----------------- TAB 3: EVIDENCE EXPLORER -----------------
with tab_evidence:
    st.subheader("🔍 Retrieval Evidence Explorer")
    st.write("Inspect the exact candidate chunks, relevance scores, and metadata retrieved for the most recent query.")

    if not st.session_state.last_query_result or not st.session_state.last_query_result.get("evidence_chunks"):
        st.info("No query has been executed yet. Ask a question in the Clinical Assistant tab to explore retrieved evidence.")
    else:
        res = st.session_state.last_query_result
        st.markdown(f"**Retrieval Strategy Used:** `{res.get('retrieval_mode', 'hybrid_rerank')}`")
        st.markdown(f"**Evidence Passages Retrieved:** `{len(res['evidence_chunks'])}`")
        
        # Citation report summary
        cr = res.get("citation_report", {})
        if cr:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Citations", cr.get("total_citations", 0))
            col2.metric("Grounded Citations", cr.get("grounded_citations", 0))
            col3.metric("Citation Accuracy", f"{cr.get('citation_accuracy', 0.0)}%")

        st.divider()

        for idx, item in enumerate(res["evidence_chunks"], start=1):
            chunk = item["chunk"]
            score = item["score"]
            score_type = item["score_type"]

            with st.expander(f"Passage #{idx}: {chunk['filename']} | Page {chunk['page_number']} (Score: {score} [{score_type}])", expanded=(idx == 1)):
                if chunk.get("section_heading"):
                    st.markdown(f"**Section Heading:** `{chunk['section_heading']}`")
                st.markdown(f"**Chunk ID:** `{chunk['chunk_id']}` | **Word Count:** `{chunk.get('word_count', 0)}`")
                st.text_area("Passage Text Content:", value=chunk["text"], height=140, key=f"chunk_txt_{idx}")

# ----------------- TAB 4: EVALUATION DASHBOARD -----------------
with tab_eval:
    st.subheader("📈 Retrieval & Evaluation Dashboard")
    st.write("Compare experimental metrics across Vector, BM25, Hybrid (RRF), and Cross-Encoder Reranker architectures on the MediRAG clinical benchmark.")

    metrics_json_path = settings.EVAL_RESULTS_DIR / "metrics_summary.json"
    chart_img_path = settings.EVAL_RESULTS_DIR / "retrieval_comparison.png"
    csv_path = settings.EVAL_RESULTS_DIR / "retrieval_comparison.csv"

    if st.button("🚀 Run / Re-Run Benchmark Experiments Now", type="primary"):
        with st.spinner("Running full evaluation suite across all 4 retrieval strategies..."):
            from evaluation.run_experiments import run_evaluation
            run_evaluation()
            st.success("Evaluation experiments successfully completed!")
            time.sleep(1)
            st.rerun()

    if metrics_json_path.exists():
        import json
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        st.markdown("### Performance Comparison Table")
        table_rows = []
        for strat, m in summary_data.items():
            name = {
                "vector": "Dense Vector (FAISS)",
                "bm25": "Sparse BM25",
                "hybrid": "Hybrid (RRF Fusion)",
                "hybrid_rerank": "Hybrid + Cross-Encoder Rerank"
            }.get(strat, strat)
            table_rows.append({
                "Strategy": name,
                "MRR": m["mrr"],
                "Hit Rate@3": m["hit_rate@3"],
                "Recall@3": m["recall@3"],
                "Precision@3": m["precision@3"],
                "Citation Accuracy": f"{m['citation_accuracy']}%",
                "Faithfulness": m["avg_faithfulness"],
                "Abstention Acc": f"{round(m['abstention_accuracy']*100, 1)}%",
                "Avg Latency": f"{m['avg_latency_sec']}s"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        if chart_img_path.exists():
            st.markdown("### Benchmark Evaluation Visualizations")
            st.image(str(chart_img_path), caption="Empirical Retrieval and Generation Metrics across Strategies", use_column_width=True)

        if csv_path.exists():
            st.markdown("### Query-Level Benchmark Data")
            df_eval = pd.read_csv(csv_path)
            st.dataframe(df_eval, use_container_width=True)
    else:
        st.info("Evaluation results not found yet. Click the button above to run the evaluation suite.")
