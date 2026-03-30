"""
Infra Runbook Recommendation Agent — Core orchestration layer.

Flow:
  User Query → ML Classification → Embedding Search → Runbook Retrieval
  → Prompt Engineering → Ollama LLM → Actionable Resolution Output
"""

import os
from embeddings import RunbookSearchEngine
from ml_models import MLModels
from ollama_client import build_prompt, generate, is_ollama_available


RUNBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runbooks")


class InfraRunbookAgent:
    """Agentic AI-powered infrastructure runbook recommendation system."""

    def __init__(self, runbooks_dir: str = RUNBOOKS_DIR, ollama_model: str = "phi3:mini"):
        self.ollama_model = ollama_model
        print("Loading embedding model and indexing runbooks...")
        self.search_engine = RunbookSearchEngine(runbooks_dir)
        print("Training ML models...")
        self.ml_models = MLModels()
        print("Agent ready.")

    def process_alert(self, alert_text: str, top_k: int = 3) -> dict:
        """
        Process an infrastructure alert end-to-end.

        Returns a dict with:
          - query: original alert text
          - classification: ML category + confidence
          - cluster_id: K-Means cluster assignment
          - severity: estimated severity (1-10)
          - retrieved_chunks: top matching runbook excerpts
          - prompt: the full prompt sent to the LLM
          - response: LLM-generated resolution
          - ollama_available: whether Ollama was reachable
        """
        # Step 1: ML Classification
        ml_analysis = self.ml_models.full_analysis(alert_text)

        # Step 2: Semantic search over runbooks (RAG retrieval)
        retrieved_chunks = self.search_engine.search(alert_text, top_k=top_k)

        # Step 3: Build prompt with context
        prompt = build_prompt(
            user_query=alert_text,
            retrieved_chunks=retrieved_chunks,
            classification=ml_analysis["classification"],
            severity=ml_analysis["severity"],
        )

        # Step 4: Call Ollama LLM
        ollama_available = is_ollama_available(self.ollama_model)
        if ollama_available:
            response = generate(prompt, model=self.ollama_model)
        else:
            response = (
                "⚠️ Ollama is not available. Make sure Ollama is running "
                "(`ollama serve`) and the phi3:mini model is pulled "
                "(`ollama pull phi3:mini`).\n\n"
                "Below is the prompt that would be sent to the LLM:\n\n"
                f"{prompt}"
            )

        return {
            "query": alert_text,
            "classification": ml_analysis["classification"],
            "cluster_id": ml_analysis["cluster_id"],
            "severity": ml_analysis["severity"],
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt,
            "response": response,
            "ollama_available": ollama_available,
        }


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = InfraRunbookAgent()

    sample_alerts = [
        "CPU usage above 90% on production VM",
        "Disk space critical on Linux server",
        "Kubernetes pod in CrashLoopBackOff",
        "Load balancer returning 502 errors",
    ]

    for alert in sample_alerts:
        print(f"\n{'='*70}")
        print(f"ALERT: {alert}")
        print("=" * 70)
        result = agent.process_alert(alert)
        print(f"Category: {result['classification']['category']} "
              f"(confidence: {result['classification']['confidence']:.0%})")
        print(f"Severity: {result['severity']}/10")
        print(f"Cluster: {result['cluster_id']}")
        print(f"\nSources: {', '.join(c['source'] for c in result['retrieved_chunks'])}")
        print(f"\n--- LLM Response ---\n{result['response']}")
