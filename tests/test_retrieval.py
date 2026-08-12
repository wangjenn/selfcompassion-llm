from pathlib import Path

import pytest

from selfcompassion_retrieval import SelfCompassionRetriever, rewrite_query

ROOT = Path(__file__).resolve().parents[1]


def test_query_rewriting_is_deterministic():
    assert rewrite_query("anxiety anxiety") == "anxiety worry stress nervous"


def test_bm25_search_returns_grounded_passages():
    retriever = SelfCompassionRetriever(ROOT / "processed_documents_clean1.json")
    response = retriever.search(
        "I keep criticizing myself for procrastinating", mode="bm25", k=5
    )
    assert response["mode"] == "bm25"
    assert len(response["results"]) == 5
    assert all(item["text"] and item["source"] for item in response["results"])
    assert all("score_bm25" in item for item in response["results"])


def test_search_rejects_blank_query():
    retriever = SelfCompassionRetriever(ROOT / "processed_documents_clean1.json")
    with pytest.raises(ValueError, match="empty"):
        retriever.search("   ")
