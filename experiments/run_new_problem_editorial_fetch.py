"""
Fetches official editorial Python solutions for the new (algorithms-category,
free, not-in-HF-snapshot) problems found by run_problem_discovery.py +
run_new_problem_metadata_fetch.py. Reuses leetcode_editorial_scraper.py's
question_solution_status/fetch_official_python_solution unchanged -- both
already take a bare slug and don't reference the HF dataframe at all, so
this is pure reuse, no new scraping logic.

Per-problem status is one of:
  "has_solution"       -- editorial exists, python3 code extracted
  "no_editorial"        -- LeetCode has no editorial for this problem at all
  "editorial_no_python"  -- editorial exists but no usable python3 approach
                            was found (rare -- e.g. approach parsing failed,
                            or no python3 variant in the chosen playground)
  "not_found"           -- slug didn't resolve at all (shouldn't happen,
                            these came from a fresh discovery pass, but
                            checked defensively)

Every attempted problem is recorded regardless of outcome -- explicitly
per this project's "flag gaps, don't just skip them" approach: a problem
with no editorial still gets its slug kept and status flagged, so a future
community-solution pass (or manual review) knows exactly what's still
missing, rather than that problem silently vanishing from view.

Writes new_problem_editorial_cache.pkl: {slug: {status, code|None}}.
Does NOT touch official_editorial_code_cache.pkl or any production
cache -- kept fully separate until explicitly reviewed and merged.

Usage:
    python run_new_problem_editorial_fetch.py [--limit N]
"""

import argparse
import pickle
import time

from leetcode_editorial_scraper import fetch_official_python_solution, question_solution_status

METADATA_PATH = "new_problem_metadata_cache.pkl"
CACHE_PATH = "new_problem_editorial_cache.pkl"
CHECKPOINT_EVERY = 20
REQUEST_DELAY_S = 0.3


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    metadata = load(METADATA_PATH, {})
    slugs = sorted(n for n, d in metadata.items() if d["status"] == "ok")
    print(f"Python-eligible new problems: {len(slugs)}")

    cache = load(CACHE_PATH, {})
    pending = [n for n in slugs if n not in cache]
    if args.limit:
        pending = pending[:args.limit]
    print(f"Processing {len(pending)} now")

    counts = {}
    for i, slug in enumerate(pending):
        try:
            status_info = question_solution_status(slug)
            if status_info is None:
                cache[slug] = {"status": "not_found", "code": None}
            elif not status_info["has_solution"] or not status_info["accessible"]:
                cache[slug] = {"status": "no_editorial", "code": None}
            else:
                code = fetch_official_python_solution(slug)
                if code:
                    cache[slug] = {"status": "has_solution", "code": code}
                else:
                    cache[slug] = {"status": "editorial_no_python", "code": None}
        except Exception as e:
            cache[slug] = {"status": f"error: {type(e).__name__}: {str(e)[:100]}", "code": None}

        s = cache[slug]["status"]
        counts[s] = counts.get(s, 0) + 1
        time.sleep(REQUEST_DELAY_S)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save(cache, CACHE_PATH)
            print(f"  ...{i + 1}/{len(pending)}  {counts}")

    save(cache, CACHE_PATH)
    print(f"\nDone. Final counts: {counts}")
    print(f"Total cached: {len(cache)}")


if __name__ == "__main__":
    main()
