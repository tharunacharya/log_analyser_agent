"""
Infra Runbook Recommendation Agent — Core orchestration layer.

Flow:
  User Query -> ML Classification -> Embedding Search -> Runbook Retrieval
  -> Prompt Engineering -> Ollama LLM -> Actionable Resolution Output
"""

import os
from embeddings import RunbookSearchEngine
from ml_models import MLModels
from ollama_client import build_prompt, generate, generate_stream, is_ollama_available


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

    def analyze(self, alert_text: str, top_k: int = 3) -> dict:
        """
        Run ML classification + RAG retrieval (fast step).
        Returns everything except the LLM response.
        """
        ml_analysis = self.ml_models.full_analysis(alert_text)
        retrieved_chunks = self.search_engine.search(alert_text, top_k=top_k)

        prompt = build_prompt(
            user_query=alert_text,
            retrieved_chunks=retrieved_chunks,
            classification=ml_analysis["classification"],
            severity=ml_analysis["severity"],
        )

        return {
            "query": alert_text,
            "classification": ml_analysis["classification"],
            "cluster_id": ml_analysis["cluster_id"],
            "severity": ml_analysis["severity"],
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt,
            "ollama_available": is_ollama_available(self.ollama_model),
        }

    def get_llm_stream(self, prompt: str):
        """Return a generator that yields LLM tokens (for streaming UI)."""
        return generate_stream(prompt, model=self.ollama_model)

    def get_llm_response(self, prompt: str) -> str:
        """Return full LLM response (blocking, for CLI)."""
        return generate(prompt, model=self.ollama_model)

    def process_alert(self, alert_text: str, top_k: int = 3) -> dict:
        """Full blocking pipeline (used by CLI)."""
        result = self.analyze(alert_text, top_k=top_k)
        if result["ollama_available"]:
            result["response"] = self.get_llm_response(result["prompt"])
        else:
            result["response"] = (
                "Ollama is not available. Run `ollama serve` and `ollama pull phi3:mini`."
            )
        return result


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
