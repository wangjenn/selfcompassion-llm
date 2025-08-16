# generate_golden.py

import json, re, os
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
DOCS_JSON = ROOT / "processed_documents_clean.json"
OUT_DIR = ROOT / "data" / "golden"
OUT_PATH = OUT_DIR / "golden_set.json"

STOP = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it","its",
    "of","on","that","the","to","was","will","with","i","you","your","we","our","they",
    "them","their","this","these","those","or","if","but","not","so","than","then","into",
    "about","over","under","after","before","between","because","while","during","against",
    "off","out","up","down","again","further","once","can","could","should","would","may",
    "might","must","do","does","did","doing","done","also"
}

def load_docs(p: Path) -> List[Dict]:
    with p.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    # normalize keys defensively
    for d in docs:
        d.setdefault("id", None)
        d.setdefault("text", "")
        d.setdefault("source", "unknown")
        d.setdefault("page_start", None)
        d.setdefault("page_end", None)
    return docs

def tokens(text: str) -> List[str]:
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    toks = [w for w in t.split() if w and w not in STOP and not w.isdigit()]
    return toks

def top_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extract frequent 2-3 word phrases (very light heuristic).
    """
    toks = tokens(text)
    bigrams  = [" ".join(toks[i:i+2]) for i in range(len(toks)-1)]
    trigrams = [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
    cand = [p for p in bigrams + trigrams if not any(w in STOP for w in p.split())]
    counts = Counter(cand)
    # prefer slightly longer phrases by boosting trigrams
    for p in list(counts):
        if len(p.split()) == 3:
            counts[p] *= 1.2
    # return unique, reasonably short phrases
    ranked = [p for p, _ in counts.most_common(20) if 6 <= len(p) <= 60]
    seen = set()
    out = []
    for p in ranked:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= max_phrases:
            break
    return out

def phrase_to_question(phrase: str) -> str:
    # simple natural question templates
    templates = [
        "What does the material say about {}?",
        "Can you explain {} in this context?",
        "What guidance is given on {}?",
        "How does the resource describe {}?",
        "What are the key points about {}?"
    ]
    # pick the first template for determinism
    return templates[0].format(phrase)

def build_golden(docs: List[Dict],
                 per_source_limit: int = 3,
                 max_total: int = 30) -> List[Dict]:
    """
    Create a small, diverse golden set across sources.
    Each question maps to the originating chunk id.
    """
    by_source = defaultdict(list)
    for d in docs:
        by_source[d.get("source","unknown")].append(d)

    items: List[Dict] = []
    for source, dd in by_source.items():
        # sort to keep determinism
        dd_sorted = sorted(dd, key=lambda x: (x.get("page_start") or 0, x.get("id") or ""))
        # sample a few evenly spaced docs per source
        take_idx = []
        n = len(dd_sorted)
        if n == 0:
            continue
        # choose up to per_source_limit positions
        step = max(1, n // per_source_limit)
        for i in range(0, n, step):
            take_idx.append(i)
            if len(take_idx) >= per_source_limit:
                break

        for i in take_idx:
            d = dd_sorted[i]
            phrases = top_phrases(d.get("text",""), max_phrases=2)
            for ph in phrases:
                q = phrase_to_question(ph)
                items.append({
                    "question": q,
                    "gold_ids": [d.get("id")],       # minimally: the doc itself
                    "note": {
                        "source": source,
                        "page_start": d.get("page_start"),
                        "page_end": d.get("page_end")
                    }
                })

    # de-duplicate questions & cap total
    deduped = []
    seen_q = set()
    for it in items:
        q = it["question"]
        if q not in seen_q and it["gold_ids"] and it["gold_ids"][0]:
            seen_q.add(q)
            deduped.append(it)
    return deduped[:max_total]

def main():
    if not DOCS_JSON.exists():
        raise FileNotFoundError(f"Missing {DOCS_JSON}. Run your ingestion first.")
    docs = load_docs(DOCS_JSON)
    golden = build_golden(docs, per_source_limit=3, max_total=30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(golden)} items → {OUT_PATH}")

if __name__ == "__main__":
    main()