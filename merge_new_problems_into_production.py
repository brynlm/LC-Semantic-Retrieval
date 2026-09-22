"""
Folds the 810 newly-scraped, newly-abstracted problems into the canonical
production caches (merged_code_only_abstract_cache.pkl /
merged_code_only_mechanism_embedding_cache.pkl) -- the same files
export_static_data.py reads, and the same override-layering pattern used
for every prior merge this project (hq_abstract_cache -> merged, then
120b/nemotron -> merged).

Full provenance is preserved on every merged abstract, not just the code
and mechanism text -- per explicit direction to be able to tell, after the
merge, which problems came from this batch and exactly how each one was
sourced:
  - new_problem: True (the only reliable way to distinguish this batch
    from the pre-existing corpus once merged -- names alone don't carry
    that information)
  - code_source: "editorial" | "community"
  - source_language: "python3" | "python" | "java" | "cpp" | "unknown"
  - anonymized: bool (False only for the ~26 non-Python entries, pending
    a real multi-language anonymizer)
  - source_verified: bool (True only for editorial -- LeetCode's own
    solution; every community entry is False regardless of vote count,
    since no execution-based check exists for problems outside the
    original HF snapshot)
  - community_meta: {topicId, title, author, votes} or None
  - batch_date: the date this whole batch was generated (this session,
    not a per-problem scrape timestamp -- the pipeline didn't record
    per-problem timing, so batch-level granularity is what's honestly
    available)

Also writes new_problem_metadata_for_export.pkl -- {name: {difficulty,
tags, description}} for the 810, since export_static_data.py's existing
pool computation and per-problem difficulty/tags/description all come
from the frozen HF dataframe (`df["name"]`), which doesn't contain these
problems at all. Without this, the 810 would silently never appear in
data.json even after this merge -- confirmed this is a real gap, not
hypothetical, earlier in this pipeline's build-out.

Usage:
    python merge_new_problems_into_production.py
"""

import datetime
import pickle

BATCH_DATE = datetime.date.today().isoformat()


def load(path, default=None):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default if default is not None else {}


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    prod_abs = load("merged_code_only_abstract_cache.pkl")
    prod_emb = load("merged_code_only_mechanism_embedding_cache.pkl")
    print(f"Production before merge: {len(prod_abs)} abstracts, {len(prod_emb)} embeddings")

    new_abs = load("new_problem_abstract_cache.pkl")
    new_emb = load("new_problem_mechanism_embedding_cache.pkl")
    new_source = load("new_problem_source_cache.pkl")
    new_meta = load("new_problem_metadata_cache.pkl")

    overlap = set(prod_abs) & set(new_abs)
    if overlap:
        raise RuntimeError(f"Unexpected overlap with production corpus: {overlap}")

    n_merged = 0
    for name, abstract in new_abs.items():
        entry = new_source[name]
        merged_entry = dict(abstract)  # keep techniques/mechanism/anonymized/source_language/source_verified as-is
        merged_entry["new_problem"] = True
        merged_entry["code_source"] = entry["source"]
        merged_entry["community_meta"] = entry.get("meta")
        merged_entry["batch_date"] = BATCH_DATE
        prod_abs[name] = merged_entry
        prod_emb[name] = new_emb[name]
        n_merged += 1

    save(prod_abs, "merged_code_only_abstract_cache.pkl")
    save(prod_emb, "merged_code_only_mechanism_embedding_cache.pkl")
    print(f"Merged {n_merged} new problems. Production after merge: {len(prod_abs)} abstracts, {len(prod_emb)} embeddings")

    # export-time metadata (difficulty/tags/description) for names not in the HF dataframe
    export_meta = {}
    for name in new_abs:
        m = new_meta[name]
        export_meta[name] = {
            "difficulty": m["difficulty"],
            "tags": m["tags"],
            "description": m["description"],
        }
    save(export_meta, "new_problem_metadata_for_export.pkl")
    print(f"Wrote new_problem_metadata_for_export.pkl: {len(export_meta)} entries")

    from collections import Counter
    print("\nProvenance breakdown of merged batch:")
    print("  code_source:", Counter(new_source[n]["source"] for n in new_abs))
    print("  source_language:", Counter(new_source[n]["language"] for n in new_abs))
    print("  anonymized:", Counter(a["anonymized"] for a in new_abs.values()))
    print("  source_verified:", Counter(new_source[n]["verified"] for n in new_abs))
    print(f"  batch_date: {BATCH_DATE}")


if __name__ == "__main__":
    main()
