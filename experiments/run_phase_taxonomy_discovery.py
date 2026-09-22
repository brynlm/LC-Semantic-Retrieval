"""
Discovery-phase batch run for the phase-taxonomy classifier prototyped in
test_phase_taxonomy.py: runs it over a tag-stratified sample (not the full
2641-problem dataset -- too early for that) so the resulting preprocessing/
algorithm labels can be eyeballed for vocabulary breadth and drift (e.g.
"prefix sum" vs "cumulative sum array" for the same concept) before doing any
canonicalization into a controlled vocabulary.

Resumable: results are cached to disk keyed by problem name, and a rerun
skips names already present in the cache -- useful since a batch this size
can hit a transient API error partway through.

Usage:
    python run_phase_taxonomy_discovery.py
"""

import os
import pickle
from collections import Counter

import categorical_similarity as cs
import test_phase_taxonomy as tpt

CACHE_PATH = os.path.join(os.path.dirname(__file__), "phase_taxonomy_discovery_cache.pkl")


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def run(per_tag_quota: int = 3, seed: int = 0) -> dict:
    df = cs.load_dataset()
    sample = cs.stratified_sample_by_tag(df, per_tag_quota=per_tag_quota, seed=seed)

    cache = load_cache()
    failures = {}
    n_new = 0

    for _, problem in sample.iterrows():
        name = problem["name"]
        if name in cache:
            continue
        try:
            cache[name] = tpt.get_phase_taxonomy(problem[["name", "description", "code"]].to_dict())
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


def summarize_vocabulary(cache: dict) -> None:
    for phase in ("preprocessing", "algorithm"):
        labels = Counter(
            entry["label"].strip().lower()
            for taxonomy in cache.values()
            for entry in taxonomy.get(phase, [])
        )
        print(f"\n=== {phase} labels: {len(labels)} distinct, {sum(labels.values())} total entries ===")
        for label, count in labels.most_common():
            print(f"{count:4d}  {label}")


if __name__ == "__main__":
    cache = run()
    summarize_vocabulary(cache)
