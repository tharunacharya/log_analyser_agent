"""
Streamlit Web UI for the Infra Runbook Recommendation Agent.
Modern UI with real-time streaming LLM responses.

Run with: streamlit run app.py --server.port 8501
"""

import streamlit as st
from agent import InfraRunbookAgent
from ollama_client import is_ollama_available

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Infra Runbook Agent",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal safe CSS — only styling, no HTML structure hacks
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
div[data-testid="stMetric"]:hover {
    border-color: #60a5fa;
}

[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b;
    border: 1px solid #334155;
    color: #e2e8f0;
    border-radius: 8px;
    font-size: 0.78rem;
    text-align: left;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155;
    border-color: #60a5fa;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialize agent (cached — loads once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models and indexing runbooks...")
def load_agent():
    return InfraRunbookAgent()

agent = load_agent()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔧 Infra Runbook Agent")
    st.caption("RAG + ML + Local LLM (phi3:mini)")
    st.divider()

    # Ollama status
    ollama_ok = is_ollama_available()
    if ollama_ok:
        st.success("✅ Ollama Connected — phi3:mini ready")
    else:
        st.error("❌ Ollama Offline")
        st.caption("Run `ollama serve` & `ollama pull phi3:mini`")

    st.divider()
    st.subheader("⚡ Quick Alerts")

    sample_alerts = [
        ("🖥️", "CPU usage above 90% on production VM"),
        ("💾", "Disk space critical on Linux server"),
        ("☸️", "Kubernetes pod in CrashLoopBackOff"),
        ("🌐", "Load balancer returning 502 errors"),
        ("☸️", "Pod OOMKilled in production namespace"),
        ("🌐", "High latency on API gateway"),
        ("💾", "No space left on device error"),
        ("🌐", "SSL certificate expiring on load balancer"),
    ]
    for icon, alert_text in sample_alerts:
        if st.button(f"{icon} {alert_text}", key=alert_text, use_container_width=True):
            st.session_state["input_alert"] = alert_text
            st.session_state["auto_run"] = True
            st.rerun()

    st.divider()
    st.subheader("🏗️ Agent Pipeline")
    st.code(
        "Alert Input\n"
        "   ↓\n"
        "ML Classification (LogReg)\n"
        "   ↓\n"
        "Embedding Search (MiniLM)\n"
        "   ↓\n"
        "Runbook Retrieval (RAG)\n"
        "   ↓\n"
        "Prompt Engineering\n"
        "   ↓\n"
        "Ollama LLM (phi3:mini)\n"
        "   ↓\n"
        "Resolution Output",
        language=None,
    )

# ---------------------------------------------------------------------------
# Main content — Header
# ---------------------------------------------------------------------------
st.markdown("# 🛠️ Infrastructure Runbook Agent")
st.markdown("**AI-powered incident resolution** — RAG retrieval + ML classification + Local LLM")
st.divider()

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
alert_input = st.text_area(
    "🚨 Alert / Incident Description",
    value=st.session_state.get("input_alert", ""),
    height=100,
    placeholder="e.g., CPU usage above 90% on production VM, Pod OOMKilled, 502 errors on load balancer...",
)

col_btn, col_clear, _ = st.columns([1.2, 0.8, 4])
with col_btn:
    run_clicked = st.button("🔍 Analyze Alert", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.pop("input_alert", None)
        st.session_state.pop("auto_run", None)
        st.rerun()

# Auto-run from sidebar click
auto_run = st.session_state.pop("auto_run", False)
should_run = (run_clicked or auto_run) and alert_input.strip()

# ---------------------------------------------------------------------------
# Process and display results
# ---------------------------------------------------------------------------
if should_run:
    query = alert_input.strip()

    # ================================================================
    # STEP 1: ML + RAG (fast)
    # ================================================================
    with st.spinner("⚙️ Running ML classification & RAG retrieval..."):
        result = agent.analyze(query)

    st.success("✅ Analysis complete — ML classification and runbook retrieval done!")

    # ================================================================
    # STEP 2: ML Insights
    # ================================================================
    st.divider()
    st.subheader("🧠 ML Insights")

    c1, c2, c3 = st.columns(3)

    cat = result["classification"]
    sev = result["severity"]

    # -- Category --
    with c1:
        st.metric(
            label="Issue Category",
            value=cat["category"],
            delta=f"{cat['confidence']:.0%} confidence",
        )

    # -- Severity --
    with c2:
        sev_label = "🔴 CRITICAL" if sev >= 8 else "🟠 HIGH" if sev >= 6 else "🟢 MEDIUM"
        st.metric(
            label="Severity Score",
            value=f"{sev} / 10",
            delta=sev_label,
            delta_color="off",
        )

    # -- Cluster --
    with c3:
        st.metric(
            label="Incident Cluster (K-Means)",
            value=f"Cluster #{result['cluster_id']}",
            delta="Similar incident group",
            delta_color="off",
        )

    # -- Category probabilities --
    st.markdown("**Classification Probabilities:**")
    prob_cols = st.columns(4)
    emoji_map = {"Compute": "🖥️", "Storage": "💾", "Network": "🌐", "Kubernetes": "☸️"}
    for i, (label, prob) in enumerate(cat["all_probabilities"].items()):
        with prob_cols[i]:
            emoji = emoji_map.get(label, "")
            st.progress(prob, text=f"{emoji} {label}: **{prob:.1%}**")

    # ================================================================
    # STEP 3: Retrieved Runbook Chunks
    # ================================================================
    st.divider()
    st.subheader("📚 Retrieved Runbook Context (RAG)")

    for i, chunk in enumerate(result["retrieved_chunks"]):
        score_pct = chunk["score"] * 100
        with st.expander(
            f"📄 **{chunk['source']}** — Match: {score_pct:.1f}%",
            expanded=(i == 0),
        ):
            st.code(chunk["text"][:800], language=None)

    # ================================================================
    # STEP 4: LLM Response (streaming)
    # ================================================================
    st.divider()
    st.subheader("🤖 AI Resolution Recommendation")

    if result["ollama_available"]:
        with st.status("🔄 Generating resolution with phi3:mini...", expanded=True) as status:
            stream = agent.get_llm_stream(result["prompt"])
            response_text = st.write_stream(stream)
            status.update(label="✅ Resolution generated!", state="complete", expanded=True)
    else:
        st.error(
            "❌ **Ollama is not available.** "
            "Start it with `ollama serve` and pull the model with `ollama pull phi3:mini`."
        )
        st.info("Showing the prompt that would be sent to the LLM:")
        with st.expander("View Prompt", expanded=True):
            st.code(result["prompt"], language=None)

    # ================================================================
    # Prompt viewer (collapsible)
    # ================================================================
    st.divider()
    with st.expander("📝 View Full Prompt Sent to LLM"):
        st.code(result["prompt"], language=None)

elif run_clicked:
    st.warning("⚠️ Please enter an alert or incident description.")

else:
    # ---- Welcome state ----
    st.divider()
    st.markdown("")
    wc1, wc2, wc3 = st.columns([1, 2, 1])
    with wc2:
        st.markdown(
            """
            <div style="text-align:center; padding: 2rem 0;">
                <div style="font-size: 4rem;">🛡️</div>
                <h3 style="color: #94a3b8; margin-top: 1rem;">Enter an alert above or pick one from the sidebar</h3>
                <p style="color: #64748b; font-size: 0.9rem;">
                    The agent will classify the issue, search relevant runbooks,<br>
                    and generate a step-by-step resolution using a local LLM.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Feature highlights
    st.markdown("")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.info("🔍 **Semantic Search**\n\nRAG over infra runbooks using cosine similarity")
    with f2:
        st.info("🧠 **ML Classification**\n\nLogistic Regression classifies Compute / Storage / Network / K8s")
    with f3:
        st.info("📊 **Clustering & Severity**\n\nK-Means groups incidents, Linear Regression estimates severity")
    with f4:
        st.info("⚡ **Local LLM**\n\nOllama phi3:mini runs entirely on your machine — no API keys needed")
