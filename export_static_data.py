"""
Exports everything the static-site explorer needs into one JSON file:
per-problem name/difficulty/tags/technique-breakdown/description/embedding,
plus a global tag->IDF-weight lookup table. Solution code is deliberately
NOT included -- the site links out to the problem's own LeetCode page
instead, which sidesteps both the legal question of redistributing
editorial/community solution code and the mismatch that comes from
generating an abstract off one implementation while a *different*
implementation gets displayed alongside it. This is the data half of the
static-site + client-side-retrieval split -- the client computes cosine
similarity (embeddings) and weighted-Dice similarity (tags) itself, so no
backend is needed for the browsing/graph-walk feature at all.

Embedding floats are rounded to 6 decimal places -- negligible precision
loss (already well within the ~0.99 cosine-similarity gap accepted between
the Python sentence-transformers embeddings and the browser's transformers.js
ONNX version used for live queries later), but meaningfully shrinks the
JSON payload versus full float32 text repr.

Also writes static_site/data-dev-mechanisms.json: a separate, small
{name: mechanism_text} file, gitignored and never deployed, that the
client merges in ONLY when explicitly running in developer mode (see
index.html's DEV_MODE check) -- lets abstracts be inspected locally for
debugging/quality-checking without them ever being part of what actually
ships to real visitors. Keeping this as a SEPARATE file rather than a flag
inside data.json matters: the production data.json genuinely never
contains the text at all, rather than containing it but hidden by a
client-side toggle (which a look at the Network tab would trivially defeat).

Usage:
    python export_static_data.py
    -> writes static_site/data.json and static_site/data-dev-mechanisms.json
"""

import json
import os
import pickle
from collections import Counter

import numpy as np

OUT_PATH = "static_site/data.json"
DEV_MECHANISMS_PATH = "static_site/data-dev-mechanisms.json"


def load(path, default=None):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default if default is not None else {}


def main():
    abstracts = load("problem_abstract_cache.pkl")
    embeddings = load("problem_embedding_cache.pkl")
    metadata = load("problem_metadata_cache.pkl")

    pool = sorted(set(abstracts) & set(embeddings) & set(metadata))
    print(f"Pool size: {len(pool)}")

    # same IDF computation the client's tag-similarity math uses internally --
    # computed once here and shipped as data, rather than recomputed per-request
    tag_sets = [set(metadata[name]["tags"]) for name in pool]
    n = len(pool)
    doc_freq = Counter(tag for tags in tag_sets for tag in tags)
    tag_idf = {tag: float(np.log(n / df_count) + 1.0) for tag, df_count in doc_freq.items()}

    problems = []
    for name in pool:
        meta = metadata[name]
        vec = np.asarray(embeddings[name], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)  # ensure unit-normalized, same invariant the client relies on
        problems.append({
            "name": name,
            "difficulty": meta["difficulty"],
            "tags": meta["tags"],
            # "mechanism" (the full LLM-generated abstract text) is deliberately
            # NOT included -- nothing in the client displays it anymore, and
            # since this whole architecture ships data.json to every visitor,
            # not including text nobody uses is a free, real privacy win (text
            # that never crosses the wire can't be copied, unlike the embedding
            # vectors it was generated from, which still need to ship for
            # client-side search to work).
            "techniques": abstracts[name].get("techniques", []),
            "description": meta["description"],
            "embedding": [round(float(x), 6) for x in vec],
        })

    payload = {"tag_idf": tag_idf, "problems": problems}

    os.makedirs("static_site", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Wrote {OUT_PATH}: {len(problems)} problems, {size_mb:.1f} MB")

    # Dev-mode-only debug info (never deployed, see module docstring) -- full
    # provenance alongside the mechanism text, so browsing in dev mode shows
    # how a given problem's code was sourced and whether it was anonymized.
    dev_mechanisms = {}
    for name in pool:
        a = abstracts[name]
        dev_mechanisms[name] = {
            "mechanism": a.get("mechanism", ""),
            "provenance": {
                "code_source": a.get("source"),
                "source_language": a.get("source_language"),
                "anonymized": a.get("anonymized"),
                "source_verified": a.get("source_verified"),
                "community_meta": a.get("community_meta"),
            },
        }
    with open(DEV_MECHANISMS_PATH, "w") as f:
        json.dump(dev_mechanisms, f)
    dev_size_mb = os.path.getsize(DEV_MECHANISMS_PATH) / (1024 * 1024)
    print(f"Wrote {DEV_MECHANISMS_PATH}: {dev_size_mb:.1f} MB (local dev only, gitignored)")


if __name__ == "__main__":
    main()
