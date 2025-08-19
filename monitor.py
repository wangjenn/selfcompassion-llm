# monitor.py
import csv, json, os, time, uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json, os
import pandas as pd
import streamlit as st

LOG_FILE = "logs/interactions.jsonl"

@st.cache_data
def load_logs():
    rows = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except:
                    pass
    return rows

rows = load_logs()
df = pd.DataFrame(rows)

st.title("Monitoring Dashboard")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

CSV_FIELDS = [
    "ts_iso","session_id","query","mode","k",
    "answer_chars","sources_count","said_idk",
    "top_sources", "top_scores_vec","top_scores_bm25","top_scores_hybrid",
    "llm_model","latency_ms","error","user_feedback","user_note"
]

def _today_csv() -> Path:
    return LOG_DIR / f"events_{datetime.utcnow().date().isoformat()}.csv"

def _append_csv(row: Dict[str, Any]):
    path = _today_csv()
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CSV_FIELDS})

def _append_jsonl(row: Dict[str, Any]):
    with (LOG_DIR / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def log_event(
    *,
    session_id: str,
    query: str,
    mode: str,
    k: int,
    answer: str,
    sources: List[tuple],
    top_scores_vec: Optional[List[float]] = None,
    top_scores_bm25: Optional[List[float]] = None,
    top_scores_hybrid: Optional[List[float]] = None,
    llm_model: str = "",
    started_at_ms: Optional[int] = None,
    error: str = "",
    user_feedback: Optional[str] = None,   # "up" | "down" | None
    user_note: Optional[str] = None
):
    now_ms = int(time.time() * 1000)
    latency_ms = now_ms - started_at_ms if started_at_ms else None
    said_idk = "i don't know" in (answer or "").lower()
    row = {
        "ts_iso": datetime.utcnow().isoformat(timespec="seconds"),
        "session_id": session_id,
        "query": query,
        "mode": mode,
        "k": k,
        "answer_chars": len(answer or ""),
        "sources_count": len(sources or []),
        "said_idk": said_idk,
        "top_sources": "; ".join([s[0] for s in (sources or [])]),
        "top_scores_vec": ",".join(f"{x:.3f}" for x in (top_scores_vec or [])) or "",
        "top_scores_bm25": ",".join(f"{x:.3f}" for x in (top_scores_bm25 or [])) or "",
        "top_scores_hybrid": ",".join(f"{x:.3f}" for x in (top_scores_hybrid or [])) or "",
        "llm_model": llm_model,
        "latency_ms": latency_ms,
        "error": error,
        "user_feedback": user_feedback or "",
        "user_note": (user_note or "").strip(),
    }
    _append_csv(row)
    _append_jsonl(row)
    

def read_last_logs(n: int = 30) -> list:
    """Return the last n log events from logs/events.jsonl as a list of dicts."""
    log_path = LOG_DIR / "events.jsonl"
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    # Parse last n lines as JSON
    return [json.loads(line) for line in lines[-n:]]

def load_saved_answers():
    """Load saved answers from logs/saved_answers.jsonl."""
    saved_file = LOG_DIR / "saved_answers.jsonl"
    if not saved_file.exists():
        return []
    
    answers = []
    with saved_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                answers.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return answers

# Load saved answers
saved_answers = load_saved_answers()
saved_df = pd.DataFrame(saved_answers) if saved_answers else pd.DataFrame()

# Display saved answers section
st.header("💾 Saved Answers")
if not saved_df.empty:
    st.write(f"Total saved answers: {len(saved_df)}")
    
    # Show recent saved answers
    st.subheader("Recent Saved Answers")
    for i, row in saved_df.tail(5).iterrows():
        with st.expander(f"Q: {row.get('query', '')[:50]}..."):
            st.write(f"**Question:** {row.get('query', '')}")
            st.write(f"**Answer:** {row.get('answer', '')[:300]}...")
            st.write(f"**Saved:** {row.get('timestamp', '')[:16]}")
            st.write(f"**Mode:** {row.get('mode', '')}")
else:
    st.write("No saved answers yet.")

st.header("LLM Evaluation Stats")

if not df.empty:
    # Prompt style breakdown
    if "prompt_style" in df.columns:
        st.subheader("Prompt Style Usage")
        st.bar_chart(df["prompt_style"].value_counts())

    # LLM model breakdown
    if "llm_model" in df.columns:
        st.subheader("LLM Model Usage")
        st.bar_chart(df["llm_model"].value_counts())

