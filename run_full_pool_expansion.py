"""
Expands code-only abstract coverage from the current pool (639/2641,
ground-truth-seeded only -- scratch_code_only_abstracts.py's main() only
ever targeted names appearing in /tmp/audit_verdicts.csv) to the full
LeetCodeDataset corpus, so the retrieval explorer and RAG-style query
feature cover as much of LeetCode's own problem set as possible.

Reuses the exact same generation function and caches as
scratch_code_only_abstracts.py (code_only_abstract_cache.pkl /
code_only_mechanism_embedding_cache.pkl) -- this is the SAME batch, just
widened to target every problem in the dataset instead of only the
audit-derived name list.

Code source: official-editorial code when available AND usable (guarded by
structural_features.is_usable_solution_code -- a handful of cached
editorial entries are incomplete prose fragments, see
scratch_structural_eval_harness.py's fix for the same issue), falling back
to the dataset's own `code` column otherwise. In practice almost every NEW
name here will fall back to the dataset's own code, since the editorial
scraper was only ever run against ground-truth-related names.

This is a much larger batch (~2000 problems) than any prior run. Expect it
to plausibly exhaust the day's rate-limit budget partway through --
get_code_only_abstract already handles that safely (once all 4 models in
MODEL_CHAIN are blacklisted for a genuine long-duration 429, every
subsequent call returns None near-instantly rather than hanging), and this
script additionally stops the whole run early once that happens rather than
churning through the remaining list generating guaranteed failures. Safely
resumable in a later run either way, since already-cached names are skipped.

Usage:
    python run_full_pool_expansion.py
"""

import pickle

from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import structural_features as sf
from scratch_code_only_abstracts import CACHE_PATH, EMB_CACHE_PATH, get_code_only_abstract

CHECKPOINT_EVERY = 20


def main():
    df = cs.load_dataset()
    name_to_code = dict(zip(df["name"], df["code"]))
    with open("official_editorial_code_cache.pkl", "rb") as f:
        official_code = pickle.load(f)

    all_names = sorted(name_to_code)
    print(f"Full dataset: {len(all_names)} problems")

    cache = {}
    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        pass
    print(f"Already cached: {len(cache)}")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    mech_emb = {}
    try:
        with open(EMB_CACHE_PATH, "rb") as f:
            mech_emb = pickle.load(f)
    except FileNotFoundError:
        pass

    remaining = [n for n in all_names if n not in cache]
    print(f"Remaining to attempt: {len(remaining)}")

    exhausted_models = set()
    n_done, n_failed = 0, 0
    stopped_early = False
    for i, name in enumerate(remaining):
        editorial = official_code.get(name)
        if editorial is not None and sf.is_usable_solution_code(editorial):
            code = editorial
        else:
            code = name_to_code[name]

        abstract = get_code_only_abstract(code, exhausted_models)
        if abstract:
            cache[name] = abstract
            mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
            n_done += 1
        else:
            n_failed += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
            with open(EMB_CACHE_PATH, "wb") as f:
                pickle.dump(mech_emb, f)
            print(f"  ...{i + 1}/{len(remaining)} processed (done={n_done}, failed={n_failed}, "
                  f"total cached={len(cache)}, exhausted_models={sorted(exhausted_models)})")
            if len(exhausted_models) >= 4:
                print("All models exhausted for the day -- stopping early rather than "
                      "churning through the remaining list generating guaranteed failures.")
                stopped_early = True
                break

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    with open(EMB_CACHE_PATH, "wb") as f:
        pickle.dump(mech_emb, f)
    pct = 100 * len(cache) / len(all_names)
    print(f"\nDone{' (stopped early -- quota exhausted)' if stopped_early else ''}. "
          f"{n_done} new, {len(cache)} total cached ({pct:.1f}% of full dataset), {n_failed} failures.")


if __name__ == "__main__":
    main()
