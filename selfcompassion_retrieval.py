"""Reusable retrieval layer for Self-Compassion AI (no Streamlit dependency)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rank_bm25 import BM25Okapi

RetrievalMode = Literal["bm25", "vector", "hybrid"]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "will", "with", "i", "you", "your", "we", "our", "they", "them", "their",
    "this", "these", "those",
}
QUERY_EXPANSIONS = {
    "anxiety": "anxiety worry stress nervous",
    "focus": "focus attention concentrate concentration",
    "overwhelmed": "overwhelmed stressed burnout overloaded",
    "criticism": "criticism harsh feedback self-talk",
    "procrastination": "procrastination delay avoidance task",
    "executive": "executive function planning organization",
    "rumination": "rumination overthinking worry thoughts",
    "self-compassion": "self-compassion kindness forgiveness acceptance",
    "email": "email communication message work",
    "work": "work job workplace professional",
}


def preprocess_text(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    return [word for word in normalized.split() if word and word not in STOPWORDS]


def rewrite_query(query: str) -> str:
    expanded: list[str] = []
    for word in query.lower().strip().split():
        expanded.append(word)
        for key, replacement in QUERY_EXPANSIONS.items():
            if key in word:
                expanded.extend(replacement.split())
                break
    return " ".join(dict.fromkeys(expanded))


def rerank_documents(
    documents: list[dict[str, Any]], query: str, rerank_top_k: int = 3
) -> list[dict[str, Any]]:
    query_terms = set(preprocess_text(query))
    scored: list[dict[str, Any]] = []
    for document in documents:
        item = dict(document)
        doc_terms = set(preprocess_text(str(item.get("text", ""))))
        item["rerank_score"] = (
            len(query_terms & doc_terms) / len(query_terms) if query_terms else 0.0
        )
        scored.append(item)
    boundary = min(rerank_top_k, len(scored))
    return sorted(
        scored[:boundary], key=lambda item: item["rerank_score"], reverse=True
    ) + scored[boundary:]


class SelfCompassionRetriever:
    """Lazy BM25/vector retriever backed by the existing corpus artifacts."""

    def __init__(
        self,
        docs_path: str | Path | None = None,
        embeddings_path: str | Path | None = None,
        index_path: str | Path | None = None,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        root = Path(__file__).resolve().parent
        self.docs_path = Path(docs_path or os.getenv(
            "SELF_COMPASSION_DOCS_PATH", root / "processed_documents_clean1.json"
        ))
        self.embeddings_path = Path(embeddings_path or os.getenv(
            "SELF_COMPASSION_EMBEDDINGS_PATH", root / "embeddings.npy"
        ))
        self.index_path = Path(index_path or os.getenv(
            "SELF_COMPASSION_INDEX_PATH", root / "id_index.json"
        ))
        self.embedding_model = embedding_model
        self._documents: list[dict[str, Any]] | None = None
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None
        self._id_order: list[str] | None = None
        self._openai_client: Any = None

    @property
    def documents(self) -> list[dict[str, Any]]:
        if self._documents is None:
            with self.docs_path.open(encoding="utf-8") as file:
                self._documents = json.load(file)
        return self._documents

    @property
    def bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            self._bm25 = BM25Okapi(
                [preprocess_text(str(document["text"])) for document in self.documents]
            )
        return self._bm25

    def _client(self) -> Any:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for vector and hybrid retrieval.")
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._openai_client

    def _load_vector_index(self) -> np.ndarray:
        if not self.embeddings_path.exists() or not self.index_path.exists():
            raise RuntimeError(
                "Vector artifacts are missing. Run ingestion.py to create embeddings.npy "
                "and id_index.json."
            )
        if self._embeddings is None:
            embeddings = np.load(self.embeddings_path).astype(np.float32)
            self._embeddings = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            )
            with self.index_path.open(encoding="utf-8") as file:
                self._id_order = json.load(file)["order"]
            if len(self._embeddings) != len(self.documents):
                raise RuntimeError("Embedding index and corpus have different lengths.")
        return self._embeddings

    def _query_vector(self, query: str) -> np.ndarray:
        response = self._client().embeddings.create(
            model=self.embedding_model, input=[query]
        )
        vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        return vector / (np.linalg.norm(vector) + 1e-8)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        mode: RetrievalMode = "bm25",
        rewrite: bool = True,
        rerank: bool = True,
        vector_weight: float = 0.6,
    ) -> dict[str, Any]:
        original_query = query.strip()
        if not original_query:
            raise ValueError("query must not be empty")
        if not 1 <= k <= 10:
            raise ValueError("k must be between 1 and 10")
        search_query = rewrite_query(original_query) if rewrite else original_query
        bm25_scores = np.asarray(
            self.bm25.get_scores(preprocess_text(search_query)), dtype=np.float32
        )
        vector_scores: np.ndarray | None = None
        if mode in {"vector", "hybrid"}:
            vector_scores = self._load_vector_index() @ self._query_vector(search_query)

        if mode == "bm25":
            final_scores = bm25_scores
        elif mode == "vector" and vector_scores is not None:
            final_scores = vector_scores
        elif mode == "hybrid" and vector_scores is not None:
            def normalize(values: np.ndarray) -> np.ndarray:
                low, high = float(values.min()), float(values.max())
                return (values - low) / (high - low + 1e-8) if high > low else np.zeros_like(values)
            final_scores = (
                vector_weight * normalize(vector_scores)
                + (1.0 - vector_weight) * normalize(bm25_scores)
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        results: list[dict[str, Any]] = []
        for index in np.argsort(final_scores)[::-1][:k]:
            document = dict(self.documents[int(index)])
            document["score_bm25"] = float(bm25_scores[index])
            document["score_vec"] = float(vector_scores[index]) if vector_scores is not None else None
            document["score_hybrid"] = float(final_scores[index]) if mode == "hybrid" else None
            results.append(document)
        if rerank:
            results = rerank_documents(results, original_query)
        return {
            "query": original_query,
            "search_query": search_query,
            "mode": mode,
            "results": results,
        }
