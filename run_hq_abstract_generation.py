"""
Generates code-only abstracts from the reconciled higher-quality source pool
(hq_source_code_cache.pkl -- official editorial preferred, verified
community solution as fallback), replacing the raw LeetCodeDataset
`completion` as the LLM's input for every problem where a better source
exists. This is the direct answer to the "Minimum Cost to Buy Apples" issue
that started this whole effort: that problem's raw-dataset solution used a
suboptimal repeated-single-source-Dijkstra approach, while its editorial
uses genuine multi-source Dijkstra -- this run regenerates its abstract (and
every other covered problem's) from that better code.

Deliberately writes to NEW, separate cache files -- hq_abstract_cache.pkl /
hq_mechanism_embedding_cache.pkl -- NOT the production code_only_abstract_cache.pkl
lineage. Merging is a distinct, later, reviewable step (same override
pattern already used for the anon_compare_* merge earlier this project):
    merged_abs = dict(raw_dataset_abs); merged_abs.update(hq_abstract_cache)

Same generation function, same anonymization step, and the same rate-limit/
resumability handling as run_full_pool_expansion.py -- see that file's
docstring for why anonymization happens before the LLM call at all.

Usage:
    python run_hq_abstract_generation.py
"""

import pickle

from sentence_transformers import SentenceTransformer

import structural_features as sf
from scratch_code_embedding_test import anonymize_code
from scratch_code_only_abstracts import get_code_only_abstract

HQ_SOURCE_PATH = "hq_source_code_cache.pkl"
CACHE_PATH = "hq_abstract_cache.pkl"
EMB_CACHE_PATH = "hq_mechanism_embedding_cache.pkl"
CHECKPOINT_EVERY = 20


def load(path, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    hq_source = load(HQ_SOURCE_PATH, {})
    print(f"Higher-quality source pool: {len(hq_source)} problems")

    cache = load(CACHE_PATH, {})
    mech_emb = load(EMB_CACHE_PATH, {})
    print(f"Already generated: {len(cache)}")

    remaining = [n for n in sorted(hq_source) if n not in cache]
    print(f"Remaining to attempt: {len(remaining)}")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    exhausted_models = set()
    n_done, n_failed, n_unusable = 0, 0, 0
    stopped_early = False
    for i, name in enumerate(remaining):
        entry = hq_source[name]
        code = entry["code"]
        if not sf.is_usable_solution_code(code):
            n_unusable += 1
            continue
        code = anonymize_code(code)

        abstract = get_code_only_abstract(code, exhausted_models)
        if abstract:
            cache[name] = abstract
            mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
            n_done += 1
        else:
            n_failed += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save(cache, CACHE_PATH)
            save(mech_emb, EMB_CACHE_PATH)
            print(f"  ...{i + 1}/{len(remaining)} processed (done={n_done}, failed={n_failed}, "
                  f"unusable={n_unusable}, total cached={len(cache)}, exhausted_models={sorted(exhausted_models)})")
            if len(exhausted_models) >= 4:
                print("All models exhausted for the day -- stopping early rather than "
                      "churning through the remaining list generating guaranteed failures.")
                stopped_early = True
                break

    save(cache, CACHE_PATH)
    save(mech_emb, EMB_CACHE_PATH)
    pct = 100 * len(cache) / len(hq_source)
    print(f"\nDone{' (stopped early -- quota exhausted)' if stopped_early else ''}. "
          f"{n_done} new, {len(cache)} total cached ({pct:.1f}% of hq source pool), "
          f"{n_failed} failures, {n_unusable} unusable source skipped.")


if __name__ == "__main__":
    main()
