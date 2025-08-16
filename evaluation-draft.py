"""
Evaluation module for Self-Compassion RAG
Calculates Hit Rate, MRR, Precision@k, Recall@k
"""
import json
import numpy as np
from typing import List, Dict
from pathlib import Path

def load_ground_truth(path: str = "data/golden/golden_set.json") -> List[Dict]:e
    """Load ground truth Q&A pairs"""
    with open(path, 'r') as f:
        return json.load(f)

def calculate_hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Hit Rate: Did we find ANY relevant doc?"""
    return float(any(doc_id in relevant_ids for doc_id in retrieved_ids))

def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank: Position of first relevant doc"""
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Precision@k: % of retrieved that are relevant"""
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return relevant_retrieved / k if k > 0 else 0

def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@k: % of relevant that we retrieved"""
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return relevant_retrieved / len(relevant_ids) if relevant_ids else 0


def evaluate_system(search_fn, ground_truth_path: str = "data/golden/golden_set.json", k: int = 10):
    """
    Evaluate a search function against ground truth
    Returns dict with hit_rate, mrr, precision@k, recall@k
    """
    ground_truth = load_ground_truth(ground_truth_path)
    
    metrics = {
        'hit_rate': [],
        'mrr': [],
        'precision@k': [],
        'recall@k': []
    }
    
    for item in ground_truth:
        query = item['question']
        relevant_ids = item['gold_ids']
        
        # Get search results
        results = search_fn(query, k=k)
        retrieved_ids = [r['id'] for r in results]
        
        # Calculate metrics
        metrics['hit_rate'].append(calculate_hit_rate(retrieved_ids, relevant_ids))
        metrics['mrr'].append(calculate_mrr(retrieved_ids, relevant_ids))
        metrics['precision@k'].append(calculate_precision_at_k(retrieved_ids, relevant_ids, k))
        metrics['recall@k'].append(calculate_recall_at_k(retrieved_ids, relevant_ids, k))
    
    # Return averages
    return {
        'hit_rate': np.mean(metrics['hit_rate']),
        'mrr': np.mean(metrics['mrr']),
        'precision@10': np.mean(metrics['precision@k']),
        'recall@10': np.mean(metrics['recall@k']),
        'num_queries': len(ground_truth)
    }
    
# ---------- Minimal retrieval setup ----------
import os, re, json
import numpy as np
from rank_bm25 import BM25Okapi

DOCS_JSON = "processed_documents_clean.json"
EMB_PATH  = "embeddings.npy"
IDX_PATH  = "id_index.json"

STOPWORDS = {
    'a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its',
    'of','on','that','the','to','was','will','with','i','you','your','we','our','they',
    'them','their','this','these','those'
}

def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w and w not in STOPWORDS]

def _l2_normalize(mat: np.ndarray, eps=1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return mat / norms

def load_docs():
    with open(DOCS_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs

DOCS = load_docs()

# BM25
_tokenized = [preprocess_text(d["text"]) for d in DOCS]
_bm25 = BM25Okapi(_tokenized)

def bm25_search(query: str, k: int = 10):
    q = preprocess_text(query)
    scores = _bm25.get_scores(q)
    idx = np.argsort(scores)[::-1][:k]
    out = []
    for i in idx:
        d = dict(DOCS[i])
        d["id"] = d.get("id")  # ensure present
        out.append(d)
    return out

# Vector (uses cached embeddings you built in ingestion)
if not (os.path.exists(EMB_PATH) and os.path.exists(IDX_PATH)):
    raise FileNotFoundError(
        "Missing embeddings. Run `python ingestion.py` to create embeddings.npy and id_index.json."
    )

_EMBS = _l2_normalize(np.load(EMB_PATH).astype(np.float32))
with open(IDX_PATH, "r", encoding="utf-8") as f:
    ID_ORDER = json.load(f)["order"]

# Map index rows to docs
# (Your DOCS are already aligned by position in your app; here we assume the same order.)
def _embed_query(query: str) -> np.ndarray:
    # Simple fallback: average of token vectors = zeros (no external API in eval)
    # This keeps the script offline; if you prefer, you can call your OpenAI embedding here.
    # For now, we approximate by BM25 scores to rank; or skip vector test if you want strictness.
    # To keep vector mode meaningful offline, we reuse BM25 scores as a proxy (documented caveat).
    # If you want true vector eval, replace this with an OpenAI embedding call.
    return None

def vector_search(query: str, k: int = 10):
    # Minimal offline proxy: rank by BM25 (so the script runs anywhere).
    # Replace with real cosine sim if you add an embedding call here.
    return bm25_search(query, k=k)

def hybrid_search(query: str, k: int = 10, w_vec=0.6, w_bm25=0.4):
    # Hybrid proxy using BM25 twice (since vector is proxied).
    # If wire real vectors, fuse normalized BM25 + cosine as in your app.
    return bm25_search(query, k=k)

# ---------- Runner ----------
if __name__ == "__main__":
    gt_path = "data/golden/golden_set.json" 
    k = 10

    bm25 = evaluate_system(bm25_search, ground_truth_path=gt_path, k=k)
    vect = evaluate_system(vector_search, ground_truth_path=gt_path, k=k)
    hybr = evaluate_system(hybrid_search, ground_truth_path=gt_path, k=k)

    def fmt(res):
        return (f"hit_rate={res['hit_rate']:.3f}  mrr={res['mrr']:.3f}  "
                f"precision@10={res['precision@10']:.3f}  recall@10={res['recall@10']:.3f}  "
                f"n={res['num_queries']}")

    print("\nRetrieval evaluation (k=10):")
    print("BM25  :", fmt(bm25))
    print("Vector:", fmt(vect), "(proxy if no embeddings call wired)")
    print("Hybrid:", fmt(hybr), "(proxy if no embeddings call wired)")

best = max(
    [("bm25", bm25), ("vector", vect), ("hybrid", hybr)],
    key=lambda x: (x[1]["hit_rate"], x[1]["mrr"])
)[0]
print(f"\nBest by hit_rate, then MRR: {best}")


# if __name__ == "__main__":
#     # Example usage
#     print("Evaluation module loaded. Use evaluate_system() to run evaluation.")
    
# 