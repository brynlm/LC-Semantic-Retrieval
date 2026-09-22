"""
Exports everything the static-site explorer needs into one JSON file:
per-problem name/difficulty/tags/abstract/description/embedding, plus
a global tag->IDF-weight lookup table. Solution code is deliberately NOT
included -- the site links out to the problem's own LeetCode page instead,
which sidesteps both the legal question of redistributing editorial/
community solution code and the mismatch that comes from generating
abstracts off higher-quality code while the raw-dataset code that used to
be displayed alongside it could be a different implementation entirely
(confirmed on coin-change: displayed code was 2D DP, the code the abstract
was actually generated from was 1D DP). This is the data half of the
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
import pickle
from collections import Counter

import numpy as np

import categorical_similarity as cs

OUT_PATH = "static_site/data.json"
DEV_MECHANISMS_PATH = "static_site/data-dev-mechanisms.json"


def load(path, default=None):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default if default is not None else {}


def main():
    df = cs.load_dataset()
    abstracts = load("merged_code_only_abstract_cache.pkl")
    embeddings = load("merged_code_only_mechanism_embedding_cache.pkl")

    # Fresh tags fetched live from LeetCode's GraphQL API (scratch_refresh_tags.py)
    # -- the local dataset snapshot was confirmed stale (515/2641 problems had
    # diffs, including entire tag categories like "Knapsack Problem"/
    # "Complete Knapsack"/"Minimax" that don't exist at all in the local
    # data). Falls back to the local tags for anything not covered.
    fresh_tags = load("fresh_tags_cache.pkl")

    # Metadata (difficulty/tags/description) for the 810 newly-scraped
    # problems -- these aren't in the frozen HF dataframe at all, so without
    # this fallback they'd silently never appear in data.json even though
    # they have real abstracts/embeddings merged in above. See
    # merge_new_problems_into_production.py for how this file is built.
    new_meta = load("new_problem_metadata_for_export.pkl")

    df_rows = {row["name"]: row for _, row in df.iterrows()}

    def row_for(name):
        if name in df_rows:
            r = df_rows[name]
            return {"difficulty": r["difficulty"], "tags": r["tags"], "description": r["description"]}
        return new_meta[name]  # KeyError here is a real bug, not something to silently paper over

    pool = sorted((set(abstracts) & set(embeddings)) & (set(df_rows) | set(new_meta)))
    n_new = sum(1 for n in pool if abstracts[n].get("new_problem"))
    print(f"Pool size: {len(pool)} ({n_new} from the newly-scraped batch)")
    print(f"Using fresh live tags for {sum(1 for n in pool if n in fresh_tags)}/{len(pool)} problems")

    def tags_for(name, fallback):
        return fresh_tags[name] if name in fresh_tags else list(fallback)

    # same IDF computation tag_similarity_matrix uses internally -- computed
    # once here and shipped as data, rather than recomputed per-request
    tag_sets = [set(tags_for(name, row_for(name)["tags"])) for name in pool]
    n = len(pool)
    doc_freq = Counter(tag for tags in tag_sets for tag in tags)
    tag_idf = {tag: float(np.log(n / df_count) + 1.0) for tag, df_count in doc_freq.items()}

    problems = []
    for name in pool:
        row = row_for(name)
        vec = np.asarray(embeddings[name], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)  # ensure unit-normalized, same invariant CODE_SIM relies on
        problems.append({
            "name": name,
            "difficulty": row["difficulty"],
            "tags": tags_for(name, row["tags"]),
            # "mechanism" (the full LLM-generated abstract text) is deliberately
            # NOT included -- nothing in the client displays it anymore, and
            # since this whole architecture ships data.json to every visitor,
            # not including text nobody uses is a free, real privacy win (see
            # the corpus-privacy discussion: text that never crosses the wire
            # can't be copied, unlike the embedding vectors it was generated
            # from, which still need to ship for client-side search to work).
            "techniques": abstracts[name].get("techniques", []),
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

    # Dev-mode-only debug info (never deployed, see module docstring). Carries
    # full provenance for the new-problem batch alongside the mechanism text,
    # so browsing in dev mode shows whether a given problem came from this
    # batch, how its code was sourced, and whether it was anonymized --
    # exactly the record explicitly asked for so manual auditing via the app
    # can distinguish these from the pre-existing corpus.
    dev_mechanisms = {}
    for name in pool:
        a = abstracts[name]
        entry = {"mechanism": a.get("mechanism", "")}
        if a.get("new_problem"):
            entry["provenance"] = {
                "new_problem": True,
                "code_source": a.get("code_source"),
                "source_language": a.get("source_language"),
                "anonymized": a.get("anonymized"),
                "source_verified": a.get("source_verified"),
                "community_meta": a.get("community_meta"),
                "batch_date": a.get("batch_date"),
            }
        dev_mechanisms[name] = entry
    with open(DEV_MECHANISMS_PATH, "w") as f:
        json.dump(dev_mechanisms, f)
    dev_size_mb = os.path.getsize(DEV_MECHANISMS_PATH) / (1024 * 1024)
    print(f"Wrote {DEV_MECHANISMS_PATH}: {dev_size_mb:.1f} MB (local dev only, gitignored)")


if __name__ == "__main__":
    main()
