"""
Extends official-editorial code fetching to the full dataset, now that a
valid Premium session is available. The original run_editorial_rollout.py
only ever covered 639 of 2641 problems (an old, smaller experimental pool);
editorial_coverage_cache.pkl's hasSolution check told us 1506 problems
actually have an editorial, so this fetches the ~1195 remaining ones that
were never attempted, reusing what's already cached (official_editorial_code_cache.pkl,
editorial_fetch_status_cache.pkl) so this is resumable and doesn't redo work.

Deliberately does NOT touch abstract generation -- this only gathers source
code. Abstract regeneration from the reconciled (editorial > community >
raw-dataset) code pool is a separate, later step.

Usage:
    python run_editorial_fetch_full.py
"""

import os
import pickle
import time
from collections import Counter

import categorical_similarity as cs
import leetcode_editorial_scraper as les

OFFICIAL_CODE_CACHE_PATH = "official_editorial_code_cache.pkl"
FETCH_STATUS_CACHE_PATH = "editorial_fetch_status_cache.pkl"
COVERAGE_CACHE_PATH = "editorial_coverage_cache.pkl"

REQUEST_DELAY_S = 0.3


def load(path, default):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    df = cs.load_dataset()
    coverage = load(COVERAGE_CACHE_PATH, {})
    official_code = load(OFFICIAL_CODE_CACHE_PATH, {})
    fetch_status = load(FETCH_STATUS_CACHE_PATH, {})

    has_editorial = [n for n, v in coverage.items() if v]
    pending = [n for n in has_editorial if n not in official_code and fetch_status.get(n) != "not_found_confirmed"]
    print(f"Problems with an editorial: {len(has_editorial)}, already fetched: {len(official_code)}, pending: {len(pending)}")

    done, failed = 0, 0
    for i, name in enumerate(pending):
        try:
            code = les.fetch_official_python_solution(name)
            if code:
                official_code[name] = code
                fetch_status[name] = "found"
                done += 1
            else:
                fetch_status[name] = "not_found_confirmed"
                failed += 1
        except Exception as e:
            fetch_status[name] = f"error: {str(e)[:100]}"
            failed += 1
        time.sleep(REQUEST_DELAY_S)

        if (i + 1) % 50 == 0:
            save(official_code, OFFICIAL_CODE_CACHE_PATH)
            save(fetch_status, FETCH_STATUS_CACHE_PATH)
            print(f"  ...{i + 1}/{len(pending)} processed (found={done}, failed={failed})")

    save(official_code, OFFICIAL_CODE_CACHE_PATH)
    save(fetch_status, FETCH_STATUS_CACHE_PATH)
    print(f"\nDone. {done} newly fetched, {failed} failed/not found. Total official code cached: {len(official_code)}")


if __name__ == "__main__":
    main()
