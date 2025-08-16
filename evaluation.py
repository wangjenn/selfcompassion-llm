"""
Evaluation for Self-Compassion RAG
- No OpenAI calls
- No embeddings.npy required
- Evaluates BM25, Vector(online), TF-IDF (offline: cosine), and Hybrid
- Metrics: HitRate, MRR, Precision@k, Recall@k
"""

import json, re, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import openai
import os
import numpy as np
from rank_bm25 import BM25Okapi

# -------------------------------
# Config / defaults
# -------------------------------
DOCS_JSON_DEFAULT = "processed_documents_clean.json"
GOLDEN_DEFAULT    = "data/golden/golden_set.json"
EMBED_MODEL       = "text-embedding-3-small"

STOPWORDS = {
    'a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its',
    'of','on','that','the','to','was','will','with','i','you','your','we','our','they',
    'them','their','this','these','those'
}

def preprocess_text(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    toks = [w for w in text.split() if w and w not in STOPWORDS]
    return toks

# -------------------------------
# Load data
# -------------------------------
def load_docs(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    # ensure id/source keys exist
    for d in docs:
        d.setdefault("id", None)
        d.setdefault("source", "(unknown)")
        d.setdefault("text", "")
    return docs

def load_golden(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        gold = json.load(f)
    # expected each item: { "question": str, "gold_ids": [str, ...] }
    return gold

# -------------------------------
# BM25 retriever
# -------------------------------
class BM25Retriever:
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.tokens = [preprocess_text(d["text"]) for d in docs]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, k: int = 10) -> List[Dict]:
        q = preprocess_text(query)
        scores = self.bm25.get_scores(q)
        idx = np.argsort(scores)[::-1][:k]
        out = []
        for i in idx:
            d = dict(self.docs[i])
            d["_bm25"] = float(scores[i])
            out.append(d)
        return out

# -------------------------------
# ONLINE: Vector (OpenAI embeddings)
class VectorRetriever:
    def __init__(self, docs: List[Dict], model: str = "text-embedding-3-small", client=None):
            self.docs = docs
            self.model = model
            self.client = client
            self.online = client is not None  # ONLINE if you pass an OpenAI client
            # self._build_offline_tfidf()
            self.online = False
            if self.online:
                self._embed_texts() # Load embeddings if online
    
# -------------------------------
# OFFLINE: TF-IDF (pure NumPy) retriever-- if no vector index available
# -------------------------------
class TfidfRetriever:
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.tok_docs = [preprocess_text(d["text"]) for d in docs]

        # Build vocab & document frequencies
        vocab = {}
        for toks in self.tok_docs:
            for t in set(toks):
                vocab[t] = vocab.get(t, 0) + 1

        self.vocab = {t:i for i, t in enumerate(sorted(vocab))}
        self.idf = self._compute_idf(vocab, len(self.tok_docs))

        # Build TF-IDF matrix (sparse as dicts per doc)
        self.doc_vecs = [self._tfidf_vector(toks) for toks in self.tok_docs]
        self.doc_norms = np.array([np.linalg.norm(v) for v in self.doc_vecs])

    def _compute_idf(self, df_map: Dict[str, int], N: int) -> np.ndarray:
        # idf = log((N+1)/(df+1)) + 1 (smooth)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for t, i in self.vocab.items():
            df = df_map.get(t, 0)
            idf[i] = np.log((N + 1) / (df + 1)) + 1.0
        return idf

    def _tfidf_vector(self, toks: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        if not toks:
            return vec
        # term frequency
        for t in toks:
            idx = self.vocab.get(t)
            if idx is not None:
                vec[idx] += 1.0
        # tf
        if vec.sum() > 0:
            vec = vec / vec.sum()
        # tf-idf
        vec = vec * self.idf
        return vec

    def search(self, query: str, k: int = 10) -> List[Dict]:
        q_vec = self._tfidf_vector(preprocess_text(query))
        q_norm = np.linalg.norm(q_vec) + 1e-8

        # cosine sim to each doc
        sims = np.zeros(len(self.doc_vecs), dtype=np.float32)
        for i, dvec in enumerate(self.doc_vecs):
            denom = self.doc_norms[i] * q_norm + 1e-8
            sims[i] = float(np.dot(dvec, q_vec) / denom)

        idx = np.argsort(sims)[::-1][:k]
        out = []
        for i in idx:
            d = dict(self.docs[i])
            d["_tfidf"] = float(sims[i])
            out.append(d)
        return out

# -------------------------------
# Hybrid fusion
# -------------------------------
def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu, sd = float(np.mean(x)), float(np.std(x) + 1e-8)
    return (x - mu) / sd if sd > 0 else np.zeros_like(x)

def fuse_lists(bm25_list: List[Dict], tfidf_list: List[Dict], k: int, w_tfidf=0.6, w_bm25=0.4) -> List[Dict]:
    # Make score arrays aligned by doc index (assume same corpus order used in retrievers)
    # Key by doc id. If any doc has missing id, fall back to its source+snippet hash.
    def doc_key(d):
        if d.get("id"): return ("id", d["id"])
        txt = (d.get("source") or "") + "|" + (d.get("text") or "")[:120]
        return ("hash", hash(txt))

    # collect scores
    scores = {}
    for d in bm25_list:
        scores.setdefault(doc_key(d), {"doc": d, "bm25": None, "tfidf": None})
        scores[doc_key(d)]["bm25"] = d.get("_bm25", 0.0)
    for d in tfidf_list:
        scores.setdefault(doc_key(d), {"doc": d, "bm25": None, "tfidf": None})
        scores[doc_key(d)]["tfidf"] = d.get("_tfidf", 0.0)

    # normalize & fuse
    bm = np.array([v["bm25"] if v["bm25"] is not None else 0.0 for v in scores.values()], dtype=np.float32)
    tf = np.array([v["tfidf"] if v["tfidf"] is not None else 0.0 for v in scores.values()], dtype=np.float32)
    bm_z, tf_z = zscore(bm), zscore(tf)
    fused = w_tfidf * tf_z + w_bm25 * bm_z

    # rank by fused
    pairs = list(scores.items())
    order = np.argsort(fused)[::-1][:k]

    out = []
    for i in order:
        key, val = pairs[i]
        d = dict(val["doc"])
        d["_hybrid"] = float(fused[i])
        out.append(d)
    return out

# -------------------------------
# Metrics
# -------------------------------
def hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    return float(any(r in relevant_ids for r in retrieved_ids))

def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    top = retrieved_ids[:k]
    rel = sum(1 for r in top if r in relevant_ids)
    return rel / k if k else 0.0

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    rel = sum(1 for r in top if r in relevant_ids)
    return rel / len(relevant_ids)

# -------------------------------
# Evaluation runner
# -------------------------------
def evaluate(search_fn, golden: List[Dict], k: int = 10) -> Dict[str, float]:
    h, r, p, rec = [], [], [], []
    for item in golden:
        q = item["question"]
        gold_ids = item["gold_ids"]

        results = search_fn(q, k=k)
        retrieved_ids = [d.get("id") for d in results]

        h.append(hit_rate(retrieved_ids, gold_ids))
        r.append(mrr(retrieved_ids, gold_ids))
        p.append(precision_at_k(retrieved_ids, gold_ids, k))
        rec.append(recall_at_k(retrieved_ids, gold_ids, k))

    return {
        "hit_rate": float(np.mean(h) if h else 0.0),
        "mrr": float(np.mean(r) if r else 0.0),
        "precision@k": float(np.mean(p) if p else 0.0),
        "recall@k": float(np.mean(rec) if rec else 0.0),
        "n": len(golden),
    }

def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval without external embeddings")
    ap.add_argument("--docs", default=DOCS_JSON_DEFAULT)
    ap.add_argument("--golden", default=GOLDEN_DEFAULT)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    # Load
    docs = load_docs(args.docs)
    golden = load_golden(args.golden)

    # Build retrievers
    bm25_ret = BM25Retriever(docs)
    tfidf_ret = TfidfRetriever(docs)
    vect_ret = VectorRetriever(docs, model=EMBED_MODEL)  # Online vector retriever (if API key set)

    # Define wrapper fns
    def bm25_fn(q, k=args.k):  return bm25_ret.search(q, k=k)
    def vect_fn(q, k=args.k):
        if vect_ret.online:
            return vect_ret.search(q, k=k)
        else:
            return tfidf_ret.search(q, k=k)
    def tfidf_fn(q, k=args.k): return tfidf_ret.search(q, k=k)
    def hybrid_fn(q, k=args.k):
        b = bm25_ret.search(q, k=k*3)   # wider pool for fusion
        t = tfidf_ret.search(q, k=k*3)
        return fuse_lists(b, t, k=k)

    # Evaluate
    bm25_res  = evaluate(bm25_fn,  golden, k=args.k)
    vect_res  = evaluate(vect_fn,  golden, k=args.k)
    tfidf_res = evaluate(tfidf_fn, golden, k=args.k)
    hybr_res  = evaluate(hybrid_fn, golden, k=args.k)

    # Print
    def fmt(m): return f"hit={m['hit_rate']:.3f}  mrr={m['mrr']:.3f}  P@{args.k}={m['precision@k']:.3f}  R@{args.k}={m['recall@k']:.3f}  n={m['n']}"
    print("\nRetrieval evaluation")
    print(f"BM25   : {fmt(bm25_res)}")
    print(f"Vector : {fmt(vect_res)}")
    print(f"TF-IDF : {fmt(tfidf_res)}")
    print(f"Hybrid : {fmt(hybr_res)}")

    # Choose best by (hit_rate, then mrr)
    best = max([("bm25", bm25_res), ("vector", vect_res),("tfidf", tfidf_res), ("hybrid", hybr_res)],
               key=lambda x: (x[1]["hit_rate"], x[1]["mrr"]))[0]
    print(f"\nBest (hit→MRR tie-break): {best}")

if __name__ == "__main__":
    main()