"""
Self-Compassion RAG Implementation Example

This script demonstrates the core components of the RAG system:
1. Document indexing (BM25 + Vector embeddings)
2. Multiple search methods (BM25, Vector, Hybrid)
3. Evaluation framework
4. QA with grounding

This is a cleaned, educational version suitable for public repositories.
"""

import os
import json
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Callable
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi


class SelfCompassionRAG:
    """
    A RAG system for self-compassion content, designed for neurodiverse individuals.
    """
    
    def __init__(self, docs_path: str = "processed_documents_clean.json"):
        """
        Initialize the RAG system.
        
        Args:
            docs_path: Path to the processed documents JSON file
        """
        # Load environment
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.embed_model = "text-embedding-3-small"
        
        # Load documents
        self.docs_path = docs_path
        self.load_documents()
        
        # Initialize indexes
        self.build_indexes()
        
        # File paths for caching
        self.emb_path = Path("embeddings.npy")
        self.idx_path = Path("id_index.json")
    
    def load_documents(self):
        """Load and validate the document corpus."""
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Missing {self.docs_path}. Run ingestion.py first.")
        
        with open(self.docs_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        
        self.texts = [d["text"] for d in self.documents]
        print(f"📚 Loaded {len(self.documents)} documents")
    
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text for BM25 indexing."""
        stopwords = set([
            'a','an','and','are','as','at','be','by','for','from','has','he',
            'in','is','it','its','of','on','that','the','to','was','will','with',
            'i','you','your','we','our','they','them','their','this','these','those'
        ])
        
        t = text.lower()
        t = re.sub(r"[^\w\s]", " ", t)
        return [w for w in t.split() if w and w not in stopwords]
    
    def build_indexes(self):
        """Build BM25 and vector indexes."""
        print("🔧 Building indexes...")
        
        # BM25 index
        preprocessed_docs = [self.preprocess_text(d["text"]) for d in self.documents]
        self.bm25_index = BM25Okapi(preprocessed_docs)
        
        # Vector embeddings
        self.build_vector_index()
        
        print("✅ Indexes built successfully")
    
    def build_vector_index(self):
        """Build or load vector embeddings with caching."""
        if self.emb_path.exists() and self.idx_path.exists():
            # Try to load cached embeddings
            embeddings = np.load(self.emb_path)
            with self.idx_path.open("r") as f:
                id_index = json.load(f)
            order = id_index["order"]
            
            if len(order) == len(self.documents):
                self.embeddings = self._l2_normalize(embeddings)
                self.id_order = order
                print("✅ Loaded cached embeddings")
                return
        
        # Compute fresh embeddings
        print(f"🔧 Computing {len(self.texts)} embeddings...")
        self.embeddings = self._embed_batch(self.texts)
        self.id_order = [d["id"] for d in self.documents]
        
        # Cache embeddings
        np.save(self.emb_path, self.embeddings)
        with self.idx_path.open("w") as f:
            json.dump({"order": self.id_order}, f, indent=2)
        
        print(f"✅ Saved embeddings → {self.emb_path.resolve()}")
    
    def _embed_batch(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Embed a batch of texts using OpenAI API."""
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.embed_model, input=batch)
            vecs.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
        return np.vstack(vecs)
    
    def _l2_normalize(self, mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """L2 normalize matrix for cosine similarity."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
        return mat / norms
    
    def bm25_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """BM25 keyword-based search."""
        tokens = self.preprocess_text(query)
        scores = self.bm25_index.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:k]
        
        results = []
        for i in idx:
            doc = self.documents[i].copy()
            doc["score_bm25"] = float(scores[i])
            results.append(doc)
        return results
    
    def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search using embeddings."""
        # Embed query
        q_vec = self.client.embeddings.create(
            model=self.embed_model, input=[query]
        ).data[0].embedding
        q_vec = np.array(q_vec, dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        
        # Compute similarities
        sims = self.embeddings @ q_vec  # cosine similarity
        idx = np.argsort(sims)[::-1][:k]
        
        results = []
        for i in idx:
            doc = self.documents[i].copy()
            doc["score_vec"] = float(sims[i])
            results.append(doc)
        return results
    
    def hybrid_search(self, query: str, k: int = 5, w_vec: float = 0.6, w_bm25: float = 0.4) -> List[Dict[str, Any]]:
        """Hybrid search combining BM25 and vector scores."""
        # Get both score arrays
        tokens = self.preprocess_text(query)
        bm25_scores = self.bm25_index.get_scores(tokens)
        
        q_vec = self.client.embeddings.create(
            model=self.embed_model, input=[query]
        ).data[0].embedding
        q_vec = np.array(q_vec, dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        vec_scores = self.embeddings @ q_vec
        
        # Min-max normalize each to [0,1]
        def _norm(x):
            x = np.asarray(x, dtype=np.float32)
            lo, hi = float(np.min(x)), float(np.max(x))
            return (x - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(x)
        
        bm25_n = _norm(bm25_scores)
        vec_n = _norm(vec_scores)
        fused = w_vec * vec_n + w_bm25 * bm25_n
        
        idx = np.argsort(fused)[::-1][:k]
        results = []
        for i in idx:
            doc = self.documents[i].copy()
            doc["score_vec"] = float(vec_scores[i])
            doc["score_bm25"] = float(bm25_scores[i])
            doc["score_hybrid"] = float(fused[i])
            results.append(doc)
        return results
    
    def search(self, query: str, mode: str = "hybrid", k: int = 5) -> List[Dict[str, Any]]:
        """Route to appropriate search method."""
        mode = mode.lower()
        if mode == "bm25":
            return self.bm25_search(query, k=k)
        elif mode == "vector":
            return self.vector_search(query, k=k)
        elif mode == "hybrid":
            return self.hybrid_search(query, k=k)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def answer_with_grounding(self, query: str, search_fn: Callable = None, k: int = 5) -> Dict[str, Any]:
        """Generate grounded answers using retrieved documents."""
        if search_fn is None:
            search_fn = self.hybrid_search
        
        docs = search_fn(query, k=k)
        if not docs:
            return {"answer": "I don't know.", "sources": []}
        
        # Create context from retrieved documents
        context = "\n\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(docs)])
        
        prompt = f"""You are a supportive, ADHD-aware coach.
Answer concisely using ONLY the context. If it's not there, say you don't know.

Context:
{context}

Question: {query}
Return an answer in 4–6 sentences, then list source filenames with page ranges.
"""
        
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return {
            "answer": resp.choices[0].message.content,
            "sources": [(d["source"], d.get("page_start"), d.get("page_end")) for d in docs]
        }
    
    def evaluate(self, golden_data: List[Dict[str, Any]], k: int = 10) -> Dict[str, Dict[str, float]]:
        """Evaluate search methods using golden dataset."""
        def evaluate_method(search_fn, name):
            hits = 0
            mrr_scores = []
            precision_scores = []
            recall_scores = []
            
            for item in golden_data:
                query = item["question"]
                relevant_ids = set(item["gold_ids"])
                
                results = search_fn(query, k=k)
                retrieved_ids = [r["id"] for r in results]
                
                # Hit Rate
                hit = any(doc_id in relevant_ids for doc_id in retrieved_ids)
                hits += int(hit)
                
                # MRR
                mrr_score = 0.0
                for rank, doc_id in enumerate(retrieved_ids, 1):
                    if doc_id in relevant_ids:
                        mrr_score = 1.0 / rank
                        break
                mrr_scores.append(mrr_score)
                
                # Precision@k
                relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
                precision_k = relevant_retrieved / k if k > 0 else 0
                precision_scores.append(precision_k)
                
                # Recall@k
                recall_k = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
                recall_scores.append(recall_k)
            
            return {
                "queries": len(golden_data),
                "hit_rate": hits / len(golden_data),
                "mrr": np.mean(mrr_scores),
                "precision@k": np.mean(precision_scores),
                "recall@k": np.mean(recall_scores)
            }
        
        # Evaluate all methods
        results = {
            "bm25": evaluate_method(self.bm25_search, "BM25"),
            "vector": evaluate_method(self.vector_search, "Vector"),
            "hybrid": evaluate_method(self.hybrid_search, "Hybrid")
        }
        
        return results


def main():
    """Example usage of the RAG system."""
    print("🧠 Self-Compassion RAG System Demo")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        rag = SelfCompassionRAG()
        
        # Test queries
        test_queries = [
            "self-compassion exercise for email anxiety",
            "how to pause rumination ADHD",
            "after harsh feedback at work, how do I respond kindly?"
        ]
        
        print("\n🔍 Testing Search Methods:")
        print("-" * 30)
        
        for query in test_queries[:1]:  # Test with first query
            print(f"\nQuery: {query}")
            
            for mode in ["bm25", "vector", "hybrid"]:
                results = rag.search(query, mode=mode, k=3)
                print(f"\n{mode.upper()} Results:")
                for i, doc in enumerate(results):
                    score = doc.get(f"score_{mode}", doc.get("score_hybrid", 0))
                    print(f"  {i+1}. {doc['source']} (score: {score:.3f})")
        
        print("\n🤖 Testing QA with Grounding:")
        print("-" * 30)
        
        # Test QA
        query = "how to practice self-compassion when feeling overwhelmed"
        answer = rag.answer_with_grounding(query, rag.hybrid_search, k=3)
        
        print(f"\nQuery: {query}")
        print(f"Answer: {answer['answer']}")
        print(f"Sources: {answer['sources'][:2]}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have a .env file with OPENAI_API_KEY")
        print("2. Run 'python ingestion.py' to generate required data files")
        print("3. Install requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
