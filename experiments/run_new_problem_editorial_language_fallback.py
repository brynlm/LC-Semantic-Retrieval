"""
Reprocesses the "editorial_no_python" bucket in new_problem_editorial_cache.pkl
using leetcode_editorial_scraper.fetch_official_solution_any_language, which
falls back through python3 -> python -> java -> cpp -> javascript -> go ->
csharp -> whatever's available, instead of requiring python3 specifically.
Confirmed on a sample that at least some of these had "python" (legacy
Python 2 tag) available and were being skipped for no real reason.

The abstract-generation LLM reads and describes the underlying mechanism
regardless of source language, so there's no reason to require Python
specifically here -- see run_new_problem_metadata_fetch.py's category-level
Python filtering for the DIFFERENT, structural distinction (SQL/pandas
problems genuinely have no algorithmic Python solution at all; this is
just "the code happens to be in Java," not a category mismatch).

Also backfills language="python3" onto every pre-existing "has_solution"
entry, since those were all fetched via the python3-only path before this
fallback existed -- keeps the "language" field meaningful across the whole
cache rather than only present on newly-reprocessed entries.

Usage:
    python run_new_problem_editorial_language_fallback.py
"""

import pickle
import time

from leetcode_editorial_scraper import fetch_official_solution_any_language

CACHE_PATH = "new_problem_editorial_cache.pkl"
REQUEST_DELAY_S = 0.3


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    for slug, d in cache.items():
        if d["status"] == "has_solution" and "language" not in d:
            d["language"] = "python3"

    no_python = [s for s, d in cache.items() if d["status"] == "editorial_no_python"]
    print(f"Reprocessing {len(no_python)} editorial_no_python cases with language fallback...")

    n_recovered, n_still_none = 0, 0
    for i, slug in enumerate(no_python):
        try:
            result = fetch_official_solution_any_language(slug)
        except Exception as e:
            print(f"  [{slug}] error: {type(e).__name__}: {str(e)[:100]}")
            result = None
        if result:
            code, lang = result
            cache[slug] = {"status": "has_solution", "code": code, "language": lang}
            n_recovered += 1
        else:
            n_still_none += 1
        time.sleep(REQUEST_DELAY_S)
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(no_python)}  recovered={n_recovered}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    from collections import Counter
    print(f"\nDone. Recovered {n_recovered}/{len(no_python)} ({n_still_none} genuinely had no code in any language).")
    print("Final status counts:", Counter(d["status"] for d in cache.values()))
    lang_counts = Counter(d.get("language") for d in cache.values() if d["status"] == "has_solution")
    print("Language breakdown of has_solution:", lang_counts)


if __name__ == "__main__":
    main()
