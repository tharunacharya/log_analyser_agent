"""
ML Models module:
- Logistic Regression: Classify alert into issue categories (Compute, Storage, Network, Kubernetes)
- K-Means Clustering: Group similar incidents
- Linear Regression: Estimate severity score (1-10)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Training data — representative alert descriptions for each category
# ---------------------------------------------------------------------------
TRAINING_ALERTS = [
    # Compute
    ("CPU usage above 90% on production VM", "Compute"),
    ("High CPU load on web server", "Compute"),
    ("VM unresponsive due to CPU spike", "Compute"),
    ("Server load average critical", "Compute"),
    ("Process consuming 100% CPU", "Compute"),
    ("EC2 instance CPU alarm triggered", "Compute"),
    ("Virtual machine performance degraded", "Compute"),
    ("CPU throttling detected on host", "Compute"),
    # Storage
    ("Disk space critical on Linux server", "Storage"),
    ("Disk usage exceeded 95% threshold", "Storage"),
    ("No space left on device error", "Storage"),
    ("Storage volume almost full", "Storage"),
    ("Log files filling up disk", "Storage"),
    ("EBS volume running out of space", "Storage"),
    ("Inode usage critical on filesystem", "Storage"),
    ("Database disk usage high", "Storage"),
    # Network
    ("Load balancer returning 502 errors", "Network"),
    ("High latency on API gateway", "Network"),
    ("Network connection timeout to backend", "Network"),
    ("SSL certificate expiring soon", "Network"),
    ("DNS resolution failure", "Network"),
    ("503 Service Unavailable from load balancer", "Network"),
    ("504 Gateway Timeout errors increasing", "Network"),
    ("Firewall blocking backend traffic", "Network"),
    # Kubernetes
    ("Kubernetes pod in CrashLoopBackOff", "Kubernetes"),
    ("Pod OOMKilled in production namespace", "Kubernetes"),
    ("Deployment rollout stuck", "Kubernetes"),
    ("K8s node not ready", "Kubernetes"),
    ("Container image pull failed", "Kubernetes"),
    ("HPA unable to scale pods", "Kubernetes"),
    ("PersistentVolumeClaim pending", "Kubernetes"),
    ("Init container failing in pod", "Kubernetes"),
]

# Severity training data: (alert text, severity score 1-10)
SEVERITY_DATA = [
    ("CPU usage above 90% on production VM", 7),
    ("Disk space critical on Linux server", 8),
    ("Load balancer returning 502 errors", 9),
    ("Kubernetes pod in CrashLoopBackOff", 7),
    ("High CPU load on web server", 6),
    ("Disk usage exceeded 95% threshold", 8),
    ("SSL certificate expiring soon", 5),
    ("DNS resolution failure", 9),
    ("Pod OOMKilled in production namespace", 8),
    ("Server load average critical", 7),
    ("No space left on device error", 9),
    ("504 Gateway Timeout errors increasing", 8),
    ("Deployment rollout stuck", 6),
    ("K8s node not ready", 9),
    ("Container image pull failed", 6),
    ("Log files filling up disk", 5),
    ("Network connection timeout to backend", 8),
    ("High latency on API gateway", 7),
    ("Process consuming 100% CPU", 7),
    ("Firewall blocking backend traffic", 8),
]

CATEGORY_LABELS = ["Compute", "Storage", "Network", "Kubernetes"]


class MLModels:
    """ML models for alert classification, clustering, and severity estimation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.severity_model = LinearRegression()
        self._train()

    def _train(self):
        """Train all ML models on the built-in training data."""
        # --- Logistic Regression (classification) ---
        texts = [t[0] for t in TRAINING_ALERTS]
        labels = [CATEGORY_LABELS.index(t[1]) for t in TRAINING_ALERTS]
        X_train = self.encoder.encode(texts, normalize_embeddings=True)
        self.classifier.fit(X_train, labels)

        # --- K-Means Clustering ---
        self.kmeans.fit(X_train)

        # --- Linear Regression (severity) ---
        sev_texts = [t[0] for t in SEVERITY_DATA]
        sev_scores = [t[1] for t in SEVERITY_DATA]
        X_sev = self.encoder.encode(sev_texts, normalize_embeddings=True)
        self.severity_model.fit(X_sev, sev_scores)

    def classify(self, query: str) -> dict:
        """Classify an alert into a category with confidence."""
        embedding = self.encoder.encode([query], normalize_embeddings=True)
        predicted_idx = self.classifier.predict(embedding)[0]
        probabilities = self.classifier.predict_proba(embedding)[0]
        return {
            "category": CATEGORY_LABELS[predicted_idx],
            "confidence": float(max(probabilities)),
            "all_probabilities": {
                CATEGORY_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            },
        }

    def get_cluster(self, query: str) -> int:
        """Return the cluster ID for a query."""
        embedding = self.encoder.encode([query], normalize_embeddings=True)
        return int(self.kmeans.predict(embedding)[0])

    def estimate_severity(self, query: str) -> float:
        """Estimate severity score (1-10) for an alert."""
        embedding = self.encoder.encode([query], normalize_embeddings=True)
        score = float(self.severity_model.predict(embedding)[0])
        return round(max(1.0, min(10.0, score)), 1)

    def full_analysis(self, query: str) -> dict:
        """Run all ML models on a query and return combined results."""
        classification = self.classify(query)
        cluster_id = self.get_cluster(query)
        severity = self.estimate_severity(query)
        return {
            "classification": classification,
            "cluster_id": cluster_id,
            "severity": severity,
        }
