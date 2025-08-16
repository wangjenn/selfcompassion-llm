#!/usr/bin/env python3
"""
Self-Compassion RAG — Ingestion (MVP)
- Denoise & redact existing processed docs
- (Optional) embed to build vector index
Artifacts produced:
  - processed_documents_clean.json
  - embeddings.npy
  - id_index.json
  - run_manifest.json
"""

import os, re, json, time, hashlib, argparse
from pathlib import Path
import numpy as np

# ---- Config (paths consistent with your notebook/app) ----
IN_JSON_DEFAULT   = "processed_documents.json"          
DOCS_JSON = "processed_documents_clean.json"
OUT_JSON_CLEAN    = "processed_documents_clean.json"
EMB_PATH          = "embeddings.npy"
IDX_PATH          = "id_index.json"
MANIFEST_PATH     = "run_manifest.json"
EMBED_MODEL       = "text-embedding-3-small"

# ---- Denoise / Redact helpers ----
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[A-Za-z0-9.-]+\.\w{2,}\b")
URL_RE   = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)

def _artifact_fresh():
    p_json, p_emb = Path(DOCS_JSON), Path(EMB_PATH)
    if p_json.exists() and p_emb.exists():
        return p_emb.stat().st_mtime >= p_json.stat().st_mtime
    return False
fresh = _artifact_fresh()

    
def scrub_contacts(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = EMAIL_RE.sub("[EMAIL]", text)
    t = URL_RE.sub("[URL]", t)
    return t

def looks_index_like(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True
    t = text.strip()
    digit_ratio = sum(ch.isdigit() for ch in t) / max(1, len(t))
    many_numbers = len(re.findall(r"\b\d{1,4}(?:[-–]\d{1,4})?\b", t)) >= 5
    many_commas  = t.count(",") >= 5
    many_caps_terms = len(re.findall(r"\b[A-Z][a-z]{2,}\b(?:,\s*)", t)) >= 5
    boilerplate = re.search(r"(all rights reserved|permission to photocopy|isbn|copyright|©|doi:)", t, re.I)
    too_short = len(t.split()) < 12
    few_periods = t.count(".") + t.count("!") + t.count("?") == 0
    return (
        digit_ratio > 0.12 and (many_numbers or many_commas)
        or many_caps_terms
        or bool(boilerplate)
        or (too_short and few_periods)
    )

def looks_toc_like(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    if any(kw in t for kw in ["table of contents", "contents", "index", "glossary"]):
        return True
    dotted_leaders = len(re.findall(r"\.{3,}\s*\d{1,4}$", text, re.M)) >= 2
    many_lines_with_endnums = len(re.findall(r"^[^\n]{3,}\s+\d{1,4}$", text, re.M)) >= 3
    return dotted_leaders or many_lines_with_endnums

def looks_contact_heavy(text: str) -> bool:
    if not isinstance(text, str):
        return False
    emails = len(EMAIL_RE.findall(text))
    urls   = len(URL_RE.findall(text))
    t = text.lower()
    contact_words = sum(kw in t for kw in ["contact", "email", "phone", "fax", "twitter", "linkedin", "facebook"])
    return (emails + urls) >= 2 or contact_words >= 2

# ---- Embedding helpers (OpenAI) ----
def _embed_texts_openai(texts, model="text-embedding-3-small", batch_size=128):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment (.env)")

    client = OpenAI(api_key=api_key)
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
    return np.vstack(vecs)

def _l2_normalize(mat: np.ndarray, eps=1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return mat / norms

# ---- Utility ----
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---- Pipeline steps ----
def load_docs(input_json: Path) -> list[dict]:
    with input_json.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    # Ensure required keys exist
    for d in docs:
        d.setdefault("id", _sha1(d.get("text","")[:512] + "|" + d.get("source","unknown")))
        d.setdefault("source", "unknown")
        d.setdefault("page_start", None)
        d.setdefault("page_end", None)
    return docs

def denoise_and_redact(docs: list[dict]) -> list[dict]:
    clean, dropped = [], 0
    for d in docs:
        txt = d.get("text", "")
        if looks_index_like(txt) or looks_toc_like(txt) or looks_contact_heavy(txt):
            dropped += 1
            continue
        dd = dict(d)
        dd["text"] = scrub_contacts(txt)
        clean.append(dd)
    kept_pct = 100 * (len(clean) / max(1, (len(clean)+dropped)))
    print(f"🧹 Denoise: kept {len(clean)} / dropped {dropped}  (kept {kept_pct:.1f}%)")
    return clean

def build_vector_index(docs: list[dict], force: bool = False):
    emb_p, idx_p = Path(EMB_PATH), Path(IDX_PATH)
    if emb_p.exists() and idx_p.exists() and not force:
        print(f"📦 Using cached embeddings → {emb_p}")
        return

    texts = [d["text"] for d in docs]
    print(f"🔧 Embedding {len(texts)} docs with {EMBED_MODEL} ...")
    embs = _embed_texts_openai(texts, model=EMBED_MODEL)
    embs = _l2_normalize(embs)
    np.save(emb_p, embs)
    save_json({"order": [d["id"] for d in docs]}, idx_p)
    print(f"✅ Saved: {emb_p} and {idx_p}")

def write_manifest(args, docs_in_count, docs_out_count):
    manifest = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_json": str(Path(args.input_json).resolve()),
        "output_json": str(Path(args.output).resolve()),
        "embeddings": str(Path(EMB_PATH).resolve()),
        "id_index": str(Path(IDX_PATH).resolve()),
        "docs_in": docs_in_count,
        "docs_out": docs_out_count,
        "embed_model": EMBED_MODEL,
        "built_embeddings": bool(Path(EMB_PATH).exists()),
    }
    save_json(manifest, Path(MANIFEST_PATH))
    print(f"📝 Wrote manifest → {MANIFEST_PATH}")

# ---- CLI ----
def main():
    p = argparse.ArgumentParser(description="Ingestion (MVP): denoise + build vector index")
    p.add_argument("--input-json", default=IN_JSON_DEFAULT, help="Input processed_documents.json")
    p.add_argument("--output", default=OUT_JSON_CLEAN, help="Output clean JSON")
    p.add_argument("--no-embed", action="store_true", help="Skip embedding/index build")
    p.add_argument("--force-embed", action="store_true", help="Rebuild embeddings even if cache exists")
    args = p.parse_args()

    in_path = Path(args.input_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}. "
                                f"Export your notebook to {IN_JSON_DEFAULT} first.")

    # Check freshness of artifacts
    fresh = _artifact_fresh()
    print(f"Artifacts fresh: {fresh}")
    if not fresh:
        print("⚠️  Embeddings may be stale. Run:  `python ingestion.py --force-embed`")

    print("🚀 Ingestion started")
    docs_raw = load_docs(in_path)
    docs_clean = denoise_and_redact(docs_raw)

    out_path = Path(args.output)
    save_json(docs_clean, out_path)
    print(f"💾 Saved clean docs → {out_path}")

    if not args.no_embed:
        build_vector_index(docs_clean, force=args.force_embed)

    write_manifest(args, len(docs_raw), len(docs_clean))
    print("✅ Ingestion completed")

if __name__ == "__main__":
    main()
