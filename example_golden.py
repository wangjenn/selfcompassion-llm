# example_golden.py

import json, random, re, argparse
from pathlib import Path

def _first_sentence(text, max_words=18):
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    s = s[0] if s else text.strip()
    words = s.split()
    return " ".join(words[:max_words])

def _nouny_query(text):
    # crude “topic-y” query maker
    text = re.sub(r"[^A-Za-z0-9 \n]", " ", text)
    words = [w for w in text.split() if 3 <= len(w) <= 20]
    return " ".join(words[:6]) or _first_sentence(text, 8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_json", default="processed_documents_clean.json")
    ap.add_argument("--out", dest="out_json", default="data/golden/golden_set.json")
    ap.add_argument("--n", type=int, default=20, help="how many items")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    docs = json.loads(in_path.read_text(encoding="utf-8"))

    # sample without replacement
    random.seed(42)
    sample = random.sample(docs, k=min(args.n, len(docs)))

    golden = []
    for d in sample:
        text = d.get("text", "")
        q = _nouny_query(text)
        golden.append({
            "question": q,
            # use your existing doc ids as gold labels
            "gold_ids": [d.get("id")]
        })

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(golden)} items → {out_path}")

if __name__ == "__main__":
    main()