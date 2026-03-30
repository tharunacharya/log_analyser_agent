# Infra Runbook Recommendation Agent

AI-powered infrastructure runbook recommendation system using **RAG + ML + Local LLM (Ollama phi3:mini)**.

## Architecture

```
User Query
   |
   v
ML Classification (Logistic Regression -> Compute/Storage/Network/Kubernetes)
   |
   v
Embedding Search (SentenceTransformer all-MiniLM-L6-v2 + Cosine Similarity)
   |
   v
Runbook Retrieval (Top-K relevant chunks from local runbook files)
   |
   v
Prompt Engineering (Structured prompt with alert + context + ML insights)
   |
   v
Ollama LLM (phi3:mini - local inference)
   |
   v
Actionable Resolution Output (Root cause, steps, commands, confidence)
```

## Project Structure

```
infra-runbook-agent/
├── runbooks/                   # Sample infrastructure runbook files
│   ├── cpu_spike_vm.txt
│   ├── disk_full_linux.txt
│   ├── k8s_pod_failure.txt
│   └── load_balancer_errors.txt
├── app.py                      # Streamlit web UI
├── agent.py                    # Core orchestration agent
├── embeddings.py               # Runbook loading, chunking, vector search
├── ml_models.py                # Logistic Regression, K-Means, Linear Regression
├── ollama_client.py            # Ollama API client
├── requirements.txt
└── README.md
```

## Setup & Run (GitHub Codespace)

### 1. Start Ollama and pull the model

```bash
# Start Ollama server (if not already running)
ollama serve &

# Pull phi3:mini model
ollama pull phi3:mini
```

### 2. Install Python dependencies

```bash
cd infra-runbook-agent
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py --server.port 8501
```

Then open the forwarded port in your browser.

### 4. (Optional) CLI mode

```bash
python agent.py
```

## ML Components

| Model               | Purpose                              | Categories                            |
|---------------------|--------------------------------------|---------------------------------------|
| Logistic Regression | Classify alert into issue type       | Compute, Storage, Network, Kubernetes |
| K-Means Clustering  | Group similar incidents              | 4 clusters                            |
| Linear Regression   | Estimate severity score              | 1-10 scale                            |

## Sample Alerts to Try

- "CPU usage above 90% on production VM"
- "Disk space critical on Linux server"
- "Kubernetes pod in CrashLoopBackOff"
- "Load balancer returning 502 errors"
- "Pod OOMKilled in production namespace"
- "SSL certificate expiring on load balancer"
- "No space left on device error"

## Tech Stack

- **LLM**: Ollama with phi3:mini (local inference)
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **ML**: scikit-learn (Logistic Regression, K-Means, Linear Regression)
- **RAG**: Cosine similarity search over chunked runbook embeddings
- **UI**: Streamlit
