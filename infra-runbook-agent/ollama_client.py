"""
Ollama API client for local LLM inference using phi3:mini.
"""

import json
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi3:mini"


def generate(prompt: str, model: str = DEFAULT_MODEL, stream: bool = False) -> str:
    """Send a prompt to Ollama and return the full response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }

    try:
        if stream:
            return _generate_stream(payload)
        else:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")
    except requests.ConnectionError:
        return (
            "ERROR: Cannot connect to Ollama. "
            "Make sure Ollama is running: `ollama serve` "
            "and the phi3:mini model is pulled: `ollama pull phi3:mini`"
        )
    except requests.Timeout:
        return "ERROR: Ollama request timed out. The model may be loading — try again."
    except Exception as e:
        return f"ERROR: Ollama request failed: {e}"


def _generate_stream(payload: dict) -> str:
    """Stream response tokens from Ollama and return concatenated text."""
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
    resp.raise_for_status()
    full_response = []
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            full_response.append(token)
            if data.get("done", False):
                break
    return "".join(full_response)


def build_prompt(user_query: str, retrieved_chunks: list[dict], classification: dict, severity: float) -> str:
    """Build the prompt for the LLM with context from RAG and ML."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {chunk['source']} | Relevance: {chunk['score']:.2f}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""You are a senior Infrastructure Engineer with expertise in cloud VMs, storage systems, networking, and Kubernetes.

Given the following infrastructure alert:
"{user_query}"

ML Classification: {classification['category']} (confidence: {classification['confidence']:.0%})
Estimated Severity: {severity}/10

The following runbook excerpts were retrieved as relevant context:

{context_text}

Based on the alert and the runbook context above, provide a structured incident response:

1. **Root Cause**: What is the most likely root cause of this alert?
2. **Step-by-Step Resolution**: Provide numbered steps to resolve this issue.
3. **Commands to Execute**: List the exact shell commands an engineer should run.
4. **Confidence Score**: Rate your confidence in this recommendation (0-100%).

Be specific, actionable, and concise. Use the runbook context to inform your response."""

    return prompt


def is_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False
