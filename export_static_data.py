"""
Exports everything the static-site explorer needs into one JSON file:
per-problem name/difficulty/tags/abstract/code/description/embedding, plus
a global tag->IDF-weight lookup table. This is the data half of the
static-site + client-side-retrieval split -- the client computes cosine
similarity (embeddings) and weighted-Dice similarity (tags) itself, the
exact same math categorical_similarity.tag_similarity_matrix and
retrieval_explorer_app.py's CODE_SIM/TAG_SIM already do, just moved to the
browser so no backend is needed for the browsing/graph-walk feature at all.

Embedding floats are rounded to 6 decimal places -- negligible precision
loss (already well within the ~0.99 cosine-similarity gap accepted between
the Python sentence-transformers embeddings and the browser's transformers.js
ONNX version used for live queries later), but meaningfully shrinks the
JSON payload versus full float32 text repr.

Usage:
    python export_static_data.py
    -> writes static_site/data.json
"""

import json
import pickle
from collections import Counter

import numpy as np

import categorical_similarity as cs

OUT_PATH = "static_site/data.json"


def main():
    df = cs.load_dataset()
    with open("merged_code_only_abstract_cache.pkl", "rb") as f:
        abstracts = pickle.load(f)
    with open("merged_code_only_mechanism_embedding_cache.pkl", "rb") as f:
        embeddings = pickle.load(f)

    pool = sorted(set(abstracts) & set(embeddings) & set(df["name"]))
    pool_df = df[df["name"].isin(set(pool))].set_index("name").loc[pool].reset_index()
    print(f"Pool size: {len(pool)}")

    # same IDF computation tag_similarity_matrix uses internally -- computed
    # once here and shipped as data, rather than recomputed per-request
    tag_sets = [set(t) for t in pool_df["tags"]]
    n = len(pool)
    doc_freq = Counter(tag for tags in tag_sets for tag in tags)
    tag_idf = {tag: float(np.log(n / df_count) + 1.0) for tag, df_count in doc_freq.items()}

    problems = []
    for _, row in pool_df.iterrows():
        name = row["name"]
        vec = np.asarray(embeddings[name], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)  # ensure unit-normalized, same invariant CODE_SIM relies on
        problems.append({
            "name": name,
            "difficulty": row["difficulty"],
            "tags": list(row["tags"]),
            # "mechanism" (the full LLM-generated abstract text) is deliberately
            # NOT included -- nothing in the client displays it anymore, and
            # since this whole architecture ships data.json to every visitor,
            # not including text nobody uses is a free, real privacy win (see
            # the corpus-privacy discussion: text that never crosses the wire
            # can't be copied, unlike the embedding vectors it was generated
            # from, which still need to ship for client-side search to work).
            "techniques": abstracts[name].get("techniques", []),
            "code": row["code"],
            "description": row["description"],
            "embedding": [round(float(x), 6) for x in vec],
        })

    payload = {"tag_idf": tag_idf, "problems": problems}

    import os
    os.makedirs("static_site", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Wrote {OUT_PATH}: {len(problems)} problems, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
