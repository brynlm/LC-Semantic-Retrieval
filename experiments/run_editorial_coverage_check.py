"""
Lightweight, full-dataset check of how many problems actually have an
official LeetCode editorial at all (question.hasSolution), independent of
whether we can currently view its content (that requires a valid Premium
session; this check doesn't). Reuses the 639 problems already checked in
editorial_fetch_status_cache.pkl and extends to the rest of the dataset.

Usage:
    python run_editorial_coverage_check.py
"""

import os
import pickle
import time

import requests

import categorical_similarity as cs

GRAPHQL_URL = "https://leetcode.com/graphql"
REQUEST_DELAY_S = 0.2
COVERAGE_CACHE_PATH = "editorial_coverage_cache.pkl"  # name -> bool (has_solution)
FETCH_STATUS_CACHE_PATH = "editorial_fetch_status_cache.pkl"  # existing, from prior work

STATUS_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) { hasSolution }
}
"""


def load(path, default):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def has_solution(name: str) -> bool:
    r = requests.post(GRAPHQL_URL, json={"query": STATUS_QUERY, "variables": {"titleSlug": name}}, timeout=15)
    data = r.json().get("data", {}).get("question")
    return bool(data and data.get("hasSolution"))


def main():
    df = cs.load_dataset()
    names = sorted(df["name"].unique())

    coverage = load(COVERAGE_CACHE_PATH, {})

    # Seed from the prior 639-problem editorial rollout: "found" or "found_but_abstract_failed"
    # both mean an editorial genuinely existed and was fetched; "not_found" means
    # the fetch attempt concluded no accessible editorial existed (may undercount
    # for paid-only problems we couldn't authenticate to at the time, but at
    # minimum "found" is a solid ground truth).
    prior_status = load(FETCH_STATUS_CACHE_PATH, {})
    for name, status in prior_status.items():
        if name not in coverage:
            coverage[name] = status.startswith("found")

    pending = [n for n in names if n not in coverage]
    print(f"Total problems: {len(names)}, seeded from prior work: {len(names) - len(pending)}, pending: {len(pending)}")

    for i, name in enumerate(pending):
        try:
            coverage[name] = has_solution(name)
        except Exception as e:
            print(f"  [{name}] error: {e}")
        time.sleep(REQUEST_DELAY_S)
        if (i + 1) % 100 == 0:
            save(coverage, COVERAGE_CACHE_PATH)
            n_true = sum(coverage.values())
            print(f"  ...{i + 1}/{len(pending)} checked (has_solution so far: {n_true}/{len(coverage)})")

    save(coverage, COVERAGE_CACHE_PATH)
    n_true = sum(coverage.values())
    print(f"\nDone. {n_true}/{len(coverage)} problems have an official editorial ({n_true/len(coverage)*100:.1f}%).")


if __name__ == "__main__":
    main()
