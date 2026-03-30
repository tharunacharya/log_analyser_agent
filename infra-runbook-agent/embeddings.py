"""
Embeddings module: Load runbooks, chunk text, encode with SentenceTransformer,
and perform cosine similarity search for RAG retrieval.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer


def load_runbooks(runbooks_dir: str) -> list[dict]:
    """Load all .txt runbook files from a directory."""
    documents = []
    for filename in sorted(os.listdir(runbooks_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(runbooks_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "filename": filename,
                "content": content,
            })
    return documents


def split_into_chunks(documents: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Split documents into overlapping text chunks for embedding."""
    chunks = []
    for doc in documents:
        text = doc["content"]
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "text": chunk_text,
                "source": doc["filename"],
            })
            start += chunk_size - overlap
    return chunks


class RunbookSearchEngine:
    """Semantic search engine over infrastructure runbooks."""

    def __init__(self, runbooks_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.runbooks_dir = runbooks_dir
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self._index()

    def _index(self):
        """Load runbooks, chunk them, and compute embeddings."""
        self.documents = load_runbooks(self.runbooks_dir)
        self.chunks = split_into_chunks(self.documents)
        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top-k most relevant chunks for a query using cosine similarity."""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        # Cosine similarity (embeddings are already L2-normalized)
        scores = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": float(scores[idx]),
            })
        return results

    def get_all_embeddings(self) -> np.ndarray:
        """Return all chunk embeddings (used by ML models)."""
        return self.embeddings

    def get_all_chunks(self) -> list[dict]:
        """Return all chunks with metadata."""
        return self.chunks

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into an embedding vector."""
        return self.model.encode([query], normalize_embeddings=True)[0]
