"""
Batch run of test_approach_abstract.py's technique/mechanism extraction.

Originally ran over a tag-stratified sample (see git history / conversation:
182 problems -> only 24 evaluable ground-truth queries, since tag-stratified
sampling doesn't guarantee a query's real relevant neighbor also gets
classified). Now defaults to evaluate_categorical_similarity's seed+neighbor
ground-truth sampling instead, which is ~5-6x more evaluable-query-efficient
per classification -- pass `names` explicitly to use a different sample (e.g.
the old tag-stratified one, for continuity with earlier runs).

Resumable: results cached to disk keyed by problem name, reruns skip names
already cached.

Usage:
    python run_abstract_discovery.py
"""

import os
import pickle

import categorical_similarity as cs
import evaluate_categorical_similarity as ev
import test_approach_abstract as taa

CACHE_PATH = os.path.join(os.path.dirname(__file__), "abstract_discovery_cache.pkl")


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def run(names: set[str] | None = None, target_new: int = 100, seed: int = 1) -> dict:
    df = cs.load_dataset()
    cache = load_cache()

    if names is None:
        gt = ev.load_similar_questions_ground_truth(df)
        names = ev.ground_truth_seed_sample(df, gt, target_new=target_new, already_cached=set(cache), seed=seed)

    sample = df[df["name"].isin(names)]
    failures = {}
    n_new = 0

    for _, problem in sample.iterrows():
        name = problem["name"]
        if name in cache:
            continue
        try:
            cache[name] = taa.get_abstract(problem[["name", "description", "code"]].to_dict())
            n_new += 1
            if n_new % 20 == 0:
                save_cache(cache)
                print(f"  ...{n_new} new results so far")
        except Exception as e:
            failures[name] = str(e)
            print(f"  [failed] {name}: {e}")

    save_cache(cache)
    print(f"\nDone. {n_new} new, {len(cache)} total cached, {len(failures)} failures.")
    if failures:
        print("Failed:", list(failures.keys()))
    return cache


if __name__ == "__main__":
    run()
