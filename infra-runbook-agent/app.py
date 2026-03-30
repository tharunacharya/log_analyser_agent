"""
Streamlit Web UI for the Infra Runbook Recommendation Agent.
Modern dark-themed UI with real-time streaming LLM responses.

Run with: streamlit run app.py --server.port 8501
"""

import streamlit as st
import time
from agent import InfraRunbookAgent
from ollama_client import is_ollama_available

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Infra Runbook Agent",
    page_icon="https://img.icons8.com/color/48/server.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for modern dark-themed UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Global ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

/* ---- Hero header ---- */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

/* ---- Metric cards ---- */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #60a5fa;
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
}
.metric-card .sub {
    font-size: 0.8rem;
    margin-top: 0.25rem;
}

/* severity colors */
.sev-critical { color: #ef4444; }
.sev-high     { color: #f59e0b; }
.sev-medium   { color: #22c55e; }

/* category badge */
.cat-compute    { color: #38bdf8; }
.cat-storage    { color: #a78bfa; }
.cat-network    { color: #fb923c; }
.cat-kubernetes { color: #34d399; }

/* ---- Section headers ---- */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.5rem 0 0.75rem 0;
}
.section-hdr .icon {
    font-size: 1.3rem;
}
.section-hdr h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0;
}

/* ---- Chunk card ---- */
.chunk-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.chunk-card .chunk-hdr {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.chunk-card .chunk-src {
    font-size: 0.8rem;
    font-weight: 600;
    color: #60a5fa;
}
.chunk-card .chunk-score {
    font-size: 0.75rem;
    font-weight: 500;
    color: #22c55e;
    background: rgba(34, 197, 94, 0.1);
    padding: 2px 8px;
    border-radius: 20px;
}
.chunk-card .chunk-txt {
    font-size: 0.82rem;
    color: #cbd5e1;
    line-height: 1.5;
    max-height: 120px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* ---- LLM response ---- */
.llm-response-box {
    background: linear-gradient(135deg, #0f172a, #1a1a2e);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 0.5rem;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b;
    border: 1px solid #334155;
    color: #e2e8f0;
    border-radius: 8px;
    font-size: 0.78rem;
    padding: 0.5rem 0.75rem;
    text-align: left;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155;
    border-color: #60a5fa;
}

/* ---- Pipeline flow ---- */
.pipeline-flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
    flex-wrap: wrap;
    margin: 0.75rem 0;
}
.pipeline-step {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.35rem 0.65rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: #94a3b8;
}
.pipeline-step.active {
    border-color: #38bdf8;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.08);
}
.pipeline-arrow {
    color: #475569;
    font-size: 0.8rem;
}

/* ---- Probability bars ---- */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.prob-label {
    font-size: 0.75rem;
    color: #94a3b8;
    min-width: 80px;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}
.prob-pct {
    font-size: 0.72rem;
    color: #cbd5e1;
    min-width: 40px;
    text-align: right;
}

/* ---- Status pill ---- */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-ok  { background: rgba(34,197,94,0.15); color: #22c55e; }
.pill-err { background: rgba(239,68,68,0.15); color: #ef4444; }

/* ---- Input area ---- */
.stTextArea textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 2px rgba(96,165,250,0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialize agent (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models and indexing runbooks...")
def load_agent():
    return InfraRunbookAgent()


agent = load_agent()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Infra Runbook Agent")
    st.caption("RAG + ML + Local LLM")
    st.markdown("---")

    # Ollama status
    ollama_ok = is_ollama_available()
    if ollama_ok:
        st.markdown('<div class="status-pill pill-ok">Connected &bull; phi3:mini</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill pill-err">Ollama Offline</div>', unsafe_allow_html=True)
        st.caption("Run `ollama serve` & `ollama pull phi3:mini`")

    st.markdown("---")
    st.markdown("##### Quick Alerts")

    sample_alerts = [
        ("CPU usage above 90% on production VM", "Compute"),
        ("Disk space critical on Linux server", "Storage"),
        ("Kubernetes pod in CrashLoopBackOff", "K8s"),
        ("Load balancer returning 502 errors", "Network"),
        ("Pod OOMKilled in production namespace", "K8s"),
        ("High latency on API gateway", "Network"),
        ("No space left on device error", "Storage"),
        ("SSL certificate expiring on load balancer", "Network"),
    ]
    for alert_text, tag in sample_alerts:
        if st.button(f"[{tag}] {alert_text}", key=alert_text, use_container_width=True):
            st.session_state["input_alert"] = alert_text
            st.session_state["auto_run"] = True
            st.rerun()

    st.markdown("---")
    st.markdown("##### Agent Pipeline")
    st.markdown("""
    ```
    Alert Input
       |
    ML Classification
       |
    Embedding Search
       |
    Runbook Retrieval
       |
    Prompt Builder
       |
    Ollama phi3:mini
       |
    Resolution Output
    ```
    """)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <h1>Infrastructure Runbook Agent</h1>
    <p>AI-powered incident resolution &mdash; RAG retrieval + ML classification + Local LLM</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline status bar
# ---------------------------------------------------------------------------
def render_pipeline(active_step: int = 0):
    steps = ["Input", "ML Classify", "RAG Search", "Retrieve", "Prompt Build", "LLM Generate", "Output"]
    html_parts = []
    for i, s in enumerate(steps):
        cls = "pipeline-step active" if i <= active_step else "pipeline-step"
        html_parts.append(f'<div class="{cls}">{s}</div>')
        if i < len(steps) - 1:
            html_parts.append('<div class="pipeline-arrow">&#8594;</div>')
    st.markdown(f'<div class="pipeline-flow">{"".join(html_parts)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
alert_input = st.text_area(
    "Describe the alert or incident",
    value=st.session_state.get("input_alert", ""),
    height=90,
    placeholder="e.g., CPU usage above 90% on production VM, Pod OOMKilled, 502 errors on load balancer...",
    label_visibility="collapsed",
)

col_btn, col_clear, _ = st.columns([1.2, 0.8, 4])
with col_btn:
    run_clicked = st.button("Analyze Alert", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.pop("input_alert", None)
        st.session_state.pop("auto_run", None)
        st.rerun()

# Auto-run if user clicked a sidebar sample
auto_run = st.session_state.pop("auto_run", False)
should_run = (run_clicked or auto_run) and alert_input.strip()

# ---------------------------------------------------------------------------
# Process and display results
# ---------------------------------------------------------------------------
if should_run:
    query = alert_input.strip()

    # Step 0: show pipeline
    pipeline_placeholder = st.empty()
    with pipeline_placeholder:
        render_pipeline(0)

    # Step 1-3: ML + RAG (fast — usually < 1 second)
    with st.spinner("Running ML classification & RAG retrieval..."):
        result = agent.analyze(query)

    with pipeline_placeholder:
        render_pipeline(4)

    # ---- ML Insights ----
    st.markdown('<div class="section-hdr"><span class="icon">&#x1f9e0;</span><h3>ML Insights</h3></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # Category card
    cat = result["classification"]
    cat_class = f"cat-{cat['category'].lower()}"
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Issue Category</div>
            <div class="value {cat_class}">{cat['category']}</div>
            <div class="sub" style="color:#94a3b8">{cat['confidence']:.0%} confidence</div>
        </div>
        """, unsafe_allow_html=True)

    # Severity card
    sev = result["severity"]
    sev_class = "sev-critical" if sev >= 8 else "sev-high" if sev >= 6 else "sev-medium"
    sev_label = "CRITICAL" if sev >= 8 else "HIGH" if sev >= 6 else "MEDIUM"
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Severity Score</div>
            <div class="value {sev_class}">{sev}/10</div>
            <div class="sub {sev_class}">{sev_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Cluster card
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Incident Cluster</div>
            <div class="value" style="color:#38bdf8">#{result['cluster_id']}</div>
            <div class="sub" style="color:#94a3b8">K-Means grouping</div>
        </div>
        """, unsafe_allow_html=True)

    # Category probability bars
    colors = {"Compute": "#38bdf8", "Storage": "#a78bfa", "Network": "#fb923c", "Kubernetes": "#34d399"}
    probs_html = ""
    for label, prob in cat["all_probabilities"].items():
        color = colors.get(label, "#60a5fa")
        pct = prob * 100
        probs_html += f"""
        <div class="prob-row">
            <div class="prob-label">{label}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <div class="prob-pct">{pct:.1f}%</div>
        </div>
        """
    st.markdown(f'<div style="margin-top:0.75rem">{probs_html}</div>', unsafe_allow_html=True)

    # ---- Retrieved Runbook Chunks ----
    st.markdown('<div class="section-hdr"><span class="icon">&#x1f4da;</span><h3>Retrieved Runbook Context (RAG)</h3></div>', unsafe_allow_html=True)

    for i, chunk in enumerate(result["retrieved_chunks"]):
        score_pct = chunk["score"] * 100
        st.markdown(f"""
        <div class="chunk-card">
            <div class="chunk-hdr">
                <span class="chunk-src">{chunk['source']}</span>
                <span class="chunk-score">{score_pct:.1f}% match</span>
            </div>
            <div class="chunk-txt">{chunk['text'][:600]}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- LLM Response (streaming) ----
    st.markdown('<div class="section-hdr"><span class="icon">&#x1f916;</span><h3>AI Resolution Recommendation</h3></div>', unsafe_allow_html=True)

    with pipeline_placeholder:
        render_pipeline(5)

    if result["ollama_available"]:
        st.markdown('<div class="llm-response-box">', unsafe_allow_html=True)
        # Stream tokens in real-time — user sees response building word by word
        stream = agent.get_llm_stream(result["prompt"])
        response_text = st.write_stream(stream)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Ollama is not available. Start it with `ollama serve` and pull the model with `ollama pull phi3:mini`.")
        with st.expander("View prompt that would be sent to LLM"):
            st.code(result["prompt"], language=None)

    with pipeline_placeholder:
        render_pipeline(6)

    # ---- Prompt expander ----
    with st.expander("View Full Prompt Sent to LLM"):
        st.code(result["prompt"], language=None)

elif run_clicked:
    st.warning("Please enter an alert or incident description.")
else:
    # Show welcome state
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #64748b;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">&#x1f6e0;&#xfe0f;</div>
        <div style="font-size: 1.1rem; font-weight: 500; color: #94a3b8;">Enter an alert above or click a sample from the sidebar</div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem;">
            The agent will classify the issue, search relevant runbooks, and generate a step-by-step resolution using a local LLM.
        </div>
    </div>
    """, unsafe_allow_html=True)
