"""
Streamlit Web UI for the Infra Runbook Recommendation Agent.

Run with: streamlit run app.py
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
)

# ---------------------------------------------------------------------------
# Initialize agent (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and indexing runbooks...")
def load_agent():
    return InfraRunbookAgent()


agent = load_agent()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Infra Runbook Agent")
    st.markdown("AI-powered infrastructure incident resolution using RAG + ML + Local LLM")
    st.divider()

    # Ollama status
    ollama_ok = is_ollama_available()
    if ollama_ok:
        st.success("Ollama: Connected (phi3:mini)")
    else:
        st.error("Ollama: Not connected")
        st.caption("Run `ollama serve` and `ollama pull phi3:mini`")

    st.divider()

    # Sample alerts for quick testing
    st.subheader("Sample Alerts")
    sample_alerts = [
        "CPU usage above 90% on production VM",
        "Disk space critical on Linux server",
        "Kubernetes pod in CrashLoopBackOff",
        "Load balancer returning 502 errors",
        "Pod OOMKilled in production namespace",
        "High latency on API gateway",
        "No space left on device error",
        "SSL certificate expiring on load balancer",
    ]
    for alert_text in sample_alerts:
        if st.button(alert_text, key=alert_text, use_container_width=True):
            st.session_state["input_alert"] = alert_text

    st.divider()
    st.subheader("Architecture")
    st.code(
        "User Query\n"
        "   ↓\n"
        "ML Classification\n"
        "   ↓\n"
        "Embedding Search (RAG)\n"
        "   ↓\n"
        "Runbook Retrieval\n"
        "   ↓\n"
        "Prompt Engineering\n"
        "   ↓\n"
        "Ollama LLM (phi3:mini)\n"
        "   ↓\n"
        "Actionable Resolution",
        language=None,
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.header("Infrastructure Runbook Recommendation Agent")
st.caption("Enter an infrastructure alert or incident description to get AI-powered resolution guidance.")

# Input
alert_input = st.text_area(
    "Alert / Incident Description",
    value=st.session_state.get("input_alert", ""),
    height=100,
    placeholder="e.g., CPU usage above 90% on production VM",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    run_clicked = st.button("Analyze Alert", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear"):
        st.session_state.pop("input_alert", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Process and display results
# ---------------------------------------------------------------------------
if run_clicked and alert_input.strip():
    with st.spinner("Analyzing alert..."):
        result = agent.process_alert(alert_input.strip())

    # --- ML Insights row ---
    st.subheader("ML Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        cat = result["classification"]
        st.metric("Issue Category", cat["category"], f"{cat['confidence']:.0%} confidence")
        # Show all category probabilities
        st.caption("Category Probabilities:")
        for label, prob in cat["all_probabilities"].items():
            st.progress(prob, text=f"{label}: {prob:.1%}")
    with col2:
        st.metric("Severity Score", f"{result['severity']}/10")
        severity = result["severity"]
        if severity >= 8:
            st.error(f"CRITICAL — Severity {severity}/10")
        elif severity >= 6:
            st.warning(f"HIGH — Severity {severity}/10")
        else:
            st.info(f"MEDIUM — Severity {severity}/10")
    with col3:
        st.metric("Incident Cluster", f"Cluster {result['cluster_id']}")
        st.caption("K-Means cluster grouping similar incidents together")

    st.divider()

    # --- Retrieved Runbook Chunks ---
    st.subheader("Retrieved Runbook Context (RAG)")
    for i, chunk in enumerate(result["retrieved_chunks"]):
        with st.expander(f"Chunk {i+1} — {chunk['source']} (score: {chunk['score']:.3f})", expanded=(i == 0)):
            st.text(chunk["text"][:1000])

    st.divider()

    # --- LLM Response ---
    st.subheader("AI Resolution Recommendation")
    if result["ollama_available"]:
        st.markdown(result["response"])
    else:
        st.warning("Ollama is not available. Showing the prompt that would be sent to the LLM.")
        st.code(result["prompt"], language=None)

    # --- Prompt (collapsible) ---
    with st.expander("View Full Prompt Sent to LLM"):
        st.code(result["prompt"], language=None)

elif run_clicked:
    st.warning("Please enter an alert description.")
