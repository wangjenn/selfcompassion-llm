# streamlit_app.py
import os
import plotly.express as px
import json
import numpy as np
import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import time, uuid, json, os
from pathlib import Path
from datetime import datetime
import pandas as pd
from pathlib import Path

LOG_PATH = Path("logs/events.jsonl")  # <- unify on events.jsonl so monitor.py sees it
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_event(record: dict):
    import time, json
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# --- Session defaults ---
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "last_docs" not in st.session_state:
    st.session_state["last_docs"] = []
if "last_event_id" not in st.session_state:
    st.session_state["last_event_id"] = None


# ───────────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIG ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Self-Compassion RAG (MVP)", page_icon="💬")
st.title("💬 Self-Compassion RAG — MVP")
st.caption("Share any thoughts, emotions, or questions you may have 💕.")


# ---------- Load env / OpenAI ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in your .env file.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Data paths ----------
DOCS_JSON = "processed_documents_clean.json"
EMB_PATH = "embeddings.npy"
IDX_PATH = "id_index.json"
EMBED_MODEL = "text-embedding-3-small"

# # ---- Monitoring config ----
# LOG_DIR = Path("logs")
# LOG_DIR.mkdir(exist_ok=True)
# LOG_PATH = LOG_DIR / "events.jsonl"

# ---------- Preprocess ----------
STOPWORDS = {
    'a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its',
    'of','on','that','the','to','was','will','with','i','you','your','we','our','they',
    'them','their','this','these','those'
}

def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w and w not in STOPWORDS]

# ---------- NEW: Query rewriting ----------
def rewrite_query(query):
    """Expand queries with domain-specific terms for better retrieval."""
    query = query.lower().strip()
    
    # ADHD and self-compassion domain expansions
    expansions = {
        'anxiety': 'anxiety worry stress nervous',
        'focus': 'focus attention concentrate concentration',
        'overwhelmed': 'overwhelmed stressed burnout overloaded',
        'criticism': 'criticism harsh feedback self-talk',
        'procrastination': 'procrastination delay avoidance task',
        'executive': 'executive function planning organization',
        'rumination': 'rumination overthinking worry thoughts',
        'self-compassion': 'self-compassion kindness forgiveness acceptance',
        'email': 'email communication message work',
        'work': 'work job workplace professional'
    }
    
    expanded_terms = []
    query_words = query.split()
    
    for word in query_words:
        expanded_terms.append(word)
        # Add expansions if word matches
        for key, expansion in expansions.items():
            if key in word:
                expanded_terms.extend(expansion.split())
                break
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for term in expanded_terms:
        if term not in seen:
            seen.add(term)
            result.append(term)
    
    return ' '.join(result)

# ---------- NEW: Document re-ranking ----------
def rerank_documents(docs, query, rerank_top_k=3):
    """Simple re-ranking using query-document term overlap."""
    if len(docs) <= rerank_top_k:
        return docs
    
    query_terms = set(preprocess_text(query))
    
    # Calculate overlap scores for all documents
    for doc in docs:
        doc_terms = set(preprocess_text(doc['text']))
        if len(query_terms) > 0:
            overlap_score = len(query_terms & doc_terms) / len(query_terms)
        else:
            overlap_score = 0.0
        doc['rerank_score'] = overlap_score
    
    # Re-rank top-k documents by overlap score
    top_docs = docs[:rerank_top_k]
    remaining_docs = docs[rerank_top_k:]
    
    # Sort top documents by rerank score
    top_docs_reranked = sorted(top_docs, key=lambda x: x['rerank_score'], reverse=True)
    
    return top_docs_reranked + remaining_docs

# ---------- Load docs ----------
@st.cache_resource
def load_docs():
    with open(DOCS_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs

DOCS = load_docs()

# ---------- BM25 ----------
@st.cache_resource
def build_bm25(docs):
    tokenized = [preprocess_text(d["text"]) for d in docs]
    return BM25Okapi(tokenized)

bm25_improved = build_bm25(DOCS)

def bm25_search(query: str, k: int = 5):
    q = preprocess_text(query)
    scores = bm25_improved.get_scores(q)
    idx = np.argsort(scores)[::-1][:k]
    out = []
    for i in idx:
        d = dict(DOCS[i])
        d["score_bm25"] = float(scores[i])
        d.setdefault("score_vec", None)
        d.setdefault("score_hybrid", None)
        out.append(d)
    return out

# ---------- Vector index ----------
def _l2_normalize(mat: np.ndarray, eps=1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return mat / norms

@st.cache_resource
def get_vector_index(docs):
    if os.path.exists(EMB_PATH) and os.path.exists(IDX_PATH):
        embs = np.load(EMB_PATH)
        with open(IDX_PATH, "r") as f:
            order = json.load(f)["order"]
        embs = _l2_normalize(embs.astype(np.float32))
        return embs, order

    # otherwise embed now (one-time)
    texts = [d["text"] for d in docs]
    vecs = []
    B = 128
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
    embs = _l2_normalize(np.vstack(vecs))
    order = [d["id"] for d in docs]
    np.save(EMB_PATH, embs)
    with open(IDX_PATH, "w") as f:
        json.dump({"order": order}, f, indent=2)
    return embs, order

EMBS, ID_ORDER = get_vector_index(DOCS)

def vector_search(query: str, k: int = 5):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q = np.array(resp.data[0].embedding, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    sims = EMBS @ q
    idx = np.argsort(sims)[::-1][:k]
    out = []
    for i in idx:
        d = dict(DOCS[i])
        d["score_vec"] = float(sims[i])
        d.setdefault("score_bm25", None)
        d.setdefault("score_hybrid", None)
        out.append(d)
    return out

def hybrid_search(query: str, k: int = 5, w_vec=0.6, w_bm25=0.4):
    # BM25
    q_tokens = preprocess_text(query)
    s_bm25 = bm25_improved.get_scores(q_tokens)
    # Vector
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q = np.array(resp.data[0].embedding, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    s_vec = EMBS @ q

    def norm(x):
        x = np.asarray(x, dtype=np.float32)
        lo, hi = float(np.min(x)), float(np.max(x))
        return (x - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(x)

    fused = w_vec * norm(s_vec) + w_bm25 * norm(s_bm25)
    idx = np.argsort(fused)[::-1][:k]
    out = []
    for i in idx:
        d = dict(DOCS[i])
        d["score_vec"] = float(s_vec[i])
        d["score_bm25"] = float(s_bm25[i])
        d["score_hybrid"] = float(fused[i])
        out.append(d)
    return out

# ---------- Grounded answer ----------
def answer_with_grounding(query, docs, temperature=0.0):
    if not docs:
        return {"answer": "I don't have enough context in this library to help yet.", "sources": []}

    context = "\n\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(docs)])
    
    prompt = f"""You are a supportive, ADHD-aware coach.
Answer concisely using ONLY the context. If it's not there, say you don't know.
Do not list sources; they will be shown separately.

Context:
{context}

Question: {query}
Return an answer in 4–6 sentences.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=temperature
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [(d.get("source","(unknown)"), d.get("page_start"), d.get("page_end")) for d in docs]
    }

# ---------- Monitoring ----------
def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

def log_event(payload: dict):
    """Append one JSON record to logs/events.jsonl"""
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print("log_event error:", e)

def read_last_logs(n=50):
    """Read last n lines from events.jsonl."""
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

def _safe_series(df, col, dtype=object):
    """Return a Series if col exists, else an empty Series of dtype."""
    return df[col] if col in df.columns else pd.Series([], dtype=dtype)

# ---------- UI ----------
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retrieval mode", ["hybrid", "bm25", "vector"], index=0)
    k = st.slider("Top‑k", 3, 10, 5)
    temp = st.slider("LLM temperature", 0.0, 0.7, 0.0, 0.1)
    enable_rewrite = st.checkbox("Enable query rewriting", value=True)
    enable_rerank = st.checkbox("Enable document re-ranking", value=True)
    
    st.markdown("---")
    st.caption("Embeddings cache")
    st.write(f"`{EMB_PATH}` present: {os.path.exists(EMB_PATH)}")
    st.write(f"`{IDX_PATH}` present: {os.path.exists(IDX_PATH)}")

query = st.text_input(
    "Ask a question",
    placeholder="e.g., self-compassion exercise for anxiety"
)

# Trigger if either the button is pressed OR Enter is hit in the input box
go = st.button("Answer") or (query and st.session_state.get("last_query") != query)

# Save last query so Enter only triggers once per new input
if query:
    st.session_state["last_query"] = query

result = None
docs = []
docs_reranked = []

if go and query and query.strip():
    # 1) Query rewriting
    original_query = query
    if enable_rewrite:
        rewritten_query = rewrite_query(query)
        if rewritten_query != original_query:
            st.info(f"**Expanded query:** {rewritten_query}")
        search_query = rewritten_query
    else:
        search_query = original_query
    
    # 2) Choose retriever
    retriever_map = {
        "bm25": bm25_search,
        "vector": vector_search,
        "hybrid": hybrid_search,
    }
    retriever = retriever_map.get(mode, hybrid_search)

    try:
        # 3) Retrieve documents
        docs = retriever(search_query, k=k)
        
        # 4) Document re-ranking
        if enable_rerank:
            docs_reranked = rerank_documents(docs, original_query, rerank_top_k=3)
            st.success("✅ Applied document re-ranking")
        else:
            docs_reranked = docs
        
        # 5) Generate answer
        result = answer_with_grounding(original_query, docs_reranked[:k], temperature=temp)
        
        # Persist for reruns (so clicking feedback doesn’t erase it)
        st.session_state["last_result"] = result
        st.session_state["last_query"] = original_query
        st.session_state["last_docs"] = docs_reranked[:k]
        
        # 6) Display results
        display_result = result or st.session_state.get("last_result")
        display_docs   = docs_reranked[:k] if result else st.session_state.get("last_docs") or []
        
        if display_result:
            st.subheader("Answer")
            st.write(result.get("answer", ""))
            
            st.subheader("Sources")
            for i, src_info in enumerate(result.get("sources", []), start=1):
                src, p1, p2 = src_info
                
                if src == "HowtoADHD.pdf":
                    src = "Jessica McCabe_(2024)-HowToADHD.pdf"
                pp = f" (pages {p1}-{p2})" if p1 is not None and p2 is not None else " (pages unknown)"
                st.markdown(f"- **[{i}]** {src}{pp}")
        
        # 7) Show retrieved passages with scores
        with st.expander("Show retrieved passages + scores"):
            for i, d in enumerate(docs_reranked, 1):
                svec = d.get("score_vec")
                sbm = d.get("score_bm25") 
                shyb = d.get("score_hybrid")
                rerank = d.get("rerank_score")
                
                parts = []
                if svec is not None: parts.append(f"vec={svec:.3f}")
                if sbm is not None: parts.append(f"bm25={sbm:.3f}")
                if shyb is not None: parts.append(f"hybrid={shyb:.3f}")
                if rerank is not None: parts.append(f"rerank={rerank:.3f}")
                
                header = f"**[{i}]** {d['source']}"
                if parts:
                    header += f"  \n*scores:* {' | '.join(parts)}"
                st.markdown(header)
                st.write(d["text"][:800] + ("..." if len(d["text"]) > 800 else ""))
        
        
        # 8) Log success (now includes the answer text and structured sources)
        retrieved_docs = docs_reranked[:k]  # the docs actually used to answer

        log_event({
            "event_id": str(uuid.uuid4()),
            "ts": _now_iso(),
            "event": "qa_answer",
            "query": original_query,
            "rewritten_query": search_query,
            "mode": mode,
            "k": k,
            "rewrite_enabled": enable_rewrite,
            "rerank_enabled": enable_rerank,

            # NEW: persist answer and sources
            "answer": result.get("answer", ""),
            "sources": [
                {
                    "id": d.get("id"),
                    "source": d.get("source"),
                    "page_start": d.get("page_start"),
                    "page_end": d.get("page_end")
                }
                for d in retrieved_docs
            ],
            "retrieved_ids": [d.get("id") for d in retrieved_docs],

            # existing metrics
            "answer_len_chars": len(result.get("answer", "")),
            "sources_count": len(result.get("sources", [])),
            "error": None,
        })
        
        qa_event_id = str(uuid.uuid4())
        log_event({
            "event_id": qa_event_id,
            "ts": _now_iso(),
            "event": "qa_answer",
            "query": original_query,
            "rewritten_query": search_query,
            "mode": mode,
            "k": k,
            "rewrite_enabled": enable_rewrite,
            "rerank_enabled": enable_rerank,
            "answer": result.get("answer", ""),
            "sources": [
                {
                    "id": d.get("id"),
                    "source": d.get("source"),
                    "page_start": d.get("page_start"),
                    "page_end": d.get("page_end")
                } for d in retrieved_docs
            ],
            "retrieved_ids": [d.get("id") for d in retrieved_docs],
            "answer_len_chars": len(result.get("answer", "")),
            "sources_count": len(result.get("sources", [])),
            "error": None,
        })
        st.session_state["last_event_id"] = qa_event_id
        

    except Exception as e:
        error_msg = str(e)
        st.error(f"Something went wrong: {error_msg}")
        log_event({
            "event_id": str(uuid.uuid4()),
            "ts": _now_iso(),
            "query": original_query,
            "mode": mode,
            "answer_len_chars": 0,
            "sources_count": 0,
            "error": error_msg,
        })

# ---------- Feedback ----------
feedback = st.radio("Was this answer helpful?", ["👍🏻 Yes", "👎🏻 No"], index=None)
if feedback and result:
    log_event({
        "event_id": str(uuid.uuid4()),
        "ts": _now_iso(),
        "event": "feedback",
        "query": query,
        "mode": mode,
        "feedback": feedback,
        "answer_len_chars": len(result.get("answer", "")),
        "sources_count": len(result.get("sources", [])),
        "error": None,
    })
    st.success("Feedback recorded. Thank you!")

# ---------- Monitoring Dashboard ----------
logs = read_last_logs(500)
df = pd.DataFrame(logs) if logs else pd.DataFrame()

# Sidebar monitoring
st.sidebar.markdown("---")
st.sidebar.header("Monitoring")
st.sidebar.write(f"Log file: `{LOG_PATH}`")
st.sidebar.write(f"Total events: {len(df)}")

if not df.empty and "mode" in df.columns:
    st.sidebar.bar_chart(_safe_series(df, "mode").value_counts().sort_index())

# Main monitoring dashboard
with st.expander("📈 Monitoring Dashboard (last 30 events)"):
    if df.empty:
        st.write("No events logged yet.")
    else:
        st.dataframe(df.tail(30), use_container_width=True)

        # Answer length over time
        st.subheader("Answer length (characters)")
        ans_len = _safe_series(df, "answer_len_chars")
        if not ans_len.empty:
            st.line_chart(ans_len.reset_index(drop=True))

        # Sources count
        st.subheader("Sources per answer")
        src_cnt = _safe_series(df, "sources_count")
        if not src_cnt.empty:
            st.bar_chart(src_cnt.value_counts().sort_index())

        # Query rewriting stats
        if "rewrite_enabled" in df.columns:
            st.subheader("Query rewriting usage")
            rewrite_counts = _safe_series(df, "rewrite_enabled").value_counts()
            st.bar_chart(rewrite_counts)

        # Re-ranking stats  
        if "rerank_enabled" in df.columns:
            st.subheader("Document re-ranking usage")
            rerank_counts = _safe_series(df, "rerank_enabled").value_counts()
            st.bar_chart(rerank_counts)
            
        
        # Feedback pie chart
        fb = _safe_series(df, "feedback")
        st.subheader("Feedback counts")
        if not fb.empty:
            fb_clean = fb.replace({"👍🏻 Yes": "Yes", "👎🏻 No": "No"}).dropna()
            fb_counts = fb_clean.value_counts().rename_axis("feedback").reset_index(name="count")
            if not fb_counts.empty:
                fig_fb = px.pie(
                    fb_counts,
                    names="feedback",
                    values="count",
                    title="Feedback breakdown",
                    hole=0.4
                )
                fig_fb.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_fb, use_container_width=True)
