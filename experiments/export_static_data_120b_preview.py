"""
Same export as export_static_data.py, except the abstract/embedding pool is
overridden with the new 120b-regenerated abstracts (hq_abstract_cache_120b.pkl
+ hq_mechanism_embedding_cache_120b.pkl, falling back to the nemotron cache
for the ~123 problems covered only there) wherever available, layered on top
of the current production merged caches for everything else.

Writes to a scratch copy of static_site/ (NOT the real static_site/data.json)
so this stays a local-only preview -- see the local-vs-Cloudflare discussion:
this batch has confirmed real regressions (e.g. coin-change losing its
coin-change-ii match) alongside real wins, so nothing here should go anywhere
near a shared/public URL yet.

Usage:
    python export_static_data_120b_preview.py
    -> writes <PREVIEW_DIR>/data.json (PREVIEW_DIR set below)
"""

import json
import os
import pickle
from collections import Counter

import numpy as np

import categorical_similarity as cs

PREVIEW_DIR = "/private/tmp/claude-501/-Users-brynleemeyer-Desktop-Personal-Projects-LC-Semantic-Retrieval/5d4aa116-601f-4fa7-801d-007c4ba0b025/scratchpad/static_site_120b_preview"
OUT_PATH = os.path.join(PREVIEW_DIR, "data.json")


def load(p):
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def main():
    df = cs.load_dataset()

    abstracts = load("merged_code_only_abstract_cache.pkl")
    embeddings = load("merged_code_only_mechanism_embedding_cache.pkl")

    # layer the new 120b batch on top, nemotron filling in what 120b didn't cover
    nemo_abs = load("hq_abstract_cache_nemotron.pkl")
    nemo_emb = load("hq_mechanism_embedding_cache_nemotron.pkl")
    for n in nemo_abs:
        abstracts[n] = nemo_abs[n]
        embeddings[n] = nemo_emb[n]

    new_abs = load("hq_abstract_cache_120b.pkl")
    new_emb = load("hq_mechanism_embedding_cache_120b.pkl")
    n_overridden = 0
    for n in new_abs:
        if n in embeddings or n in abstracts:
            n_overridden += 1
        abstracts[n] = new_abs[n]
        embeddings[n] = new_emb[n]

    print(f"Overrode {n_overridden} problems with new 120b abstracts (+{len(nemo_abs)} from nemotron)")

    pool = sorted(set(abstracts) & set(embeddings) & set(df["name"]))
    pool_df = df[df["name"].isin(set(pool))].set_index("name").loc[pool].reset_index()
    print(f"Pool size: {len(pool)}")

    # Fresh tags fetched live from LeetCode's GraphQL API (scratch_refresh_tags.py)
    # -- our local snapshot was confirmed stale (515/2641 problems had diffs,
    # including newly-live "Knapsack Problem"/"Complete Knapsack"/"Minimax" tags
    # that don't exist at all in the local dataset). Falls back to the local
    # tags for anything not covered by the refresh.
    fresh_tags = load("/tmp/tag_refresh_fresh_tags.pkl")
    n_fresh_used = sum(1 for n in pool if n in fresh_tags)
    print(f"Using fresh live tags for {n_fresh_used}/{len(pool)} problems")

    def tags_for(name, fallback):
        return fresh_tags[name] if name in fresh_tags else list(fallback)

    tag_sets = [set(tags_for(row["name"], row["tags"])) for _, row in pool_df.iterrows()]
    n = len(pool)
    doc_freq = Counter(tag for tags in tag_sets for tag in tags)
    tag_idf = {tag: float(np.log(n / df_count) + 1.0) for tag, df_count in doc_freq.items()}

    problems = []
    for _, row in pool_df.iterrows():
        name = row["name"]
        vec = np.asarray(embeddings[name], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        problems.append({
            "name": name,
            "difficulty": row["difficulty"],
            "tags": tags_for(name, row["tags"]),
            "techniques": abstracts[name].get("techniques", []),
            "description": row["description"],
            "embedding": [round(float(x), 6) for x in vec],
        })

    payload = {"tag_idf": tag_idf, "problems": problems}

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Wrote {OUT_PATH}: {len(problems)} problems, {size_mb:.1f} MB")

    dev_mechanisms = {name: abstracts[name].get("mechanism", "") for name in pool}
    dev_path = os.path.join(PREVIEW_DIR, "data-dev-mechanisms.json")
    with open(dev_path, "w") as f:
        json.dump(dev_mechanisms, f)
    print(f"Wrote {dev_path}: {os.path.getsize(dev_path)/(1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
