"""
Fetches official editorial solutions for every Python-eligible problem in
problem_metadata_cache.pkl. Reuses leetcode_editorial_scraper.py's
question_solution_status/fetch_official_python_solution/
fetch_official_solution_any_language unchanged -- all three already take a
bare slug and hit LeetCode's live API directly, no dataset dependency.

Two passes, run in order by main():

1. Python3-only pass (fetch_python_editorials): per-problem status is one of
     "has_solution"        -- editorial exists, python3 code extracted
     "no_editorial"         -- LeetCode has no editorial for this problem
     "editorial_no_python"  -- editorial exists but no usable python3
                               approach was found
     "not_found"            -- slug didn't resolve at all
   Every attempted problem is recorded regardless of outcome, so a problem
   with no editorial still gets its slug kept and status flagged rather than
   silently vanishing from view.

2. Any-language fallback pass (fetch_language_fallback), over just the
   "editorial_no_python" bucket left by pass 1: falls back through
   python3 -> python -> java -> cpp -> javascript -> go -> csharp ->
   whatever's available, instead of requiring python3 specifically. The
   abstract-generation LLM reads and describes the underlying mechanism
   regardless of source language, so there's no reason to require Python
   specifically here (that's a different, narrower filter -- see
   fetch_problem_metadata.py's category-level Python check, which excludes
   genuine category mismatches like SQL/pandas problems).

Writes problem_editorial_cache.pkl: {slug: {status, code|None, language}}.

Usage:
    python fetch_problem_editorials.py [--limit N]
"""

import argparse
import pickle
import time
from collections import Counter

from leetcode_editorial_scraper import (
    fetch_official_python_solution,
    fetch_official_solution_any_language,
    question_solution_status,
)

METADATA_PATH = "problem_metadata_cache.pkl"
CACHE_PATH = "problem_editorial_cache.pkl"
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


def fetch_python_editorials(limit=None):
    metadata = load(METADATA_PATH, {})
    slugs = sorted(n for n, d in metadata.items() if d["status"] == "ok")
    print(f"Python-eligible problems: {len(slugs)}")

    cache = load(CACHE_PATH, {})
    pending = [n for n in slugs if n not in cache]
    if limit:
        pending = pending[:limit]
    print(f"Processing {len(pending)} now (python3-only pass)")

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
                    cache[slug] = {"status": "has_solution", "code": code, "language": "python3"}
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
    print(f"Python3-only pass done. Counts this run: {counts}")


def fetch_language_fallback(limit=None):
    cache = load(CACHE_PATH, {})

    for slug, d in cache.items():
        if d["status"] == "has_solution" and "language" not in d:
            d["language"] = "python3"

    no_python = [s for s, d in cache.items() if d["status"] == "editorial_no_python"]
    if limit:
        no_python = no_python[:limit]
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

    save(cache, CACHE_PATH)
    print(f"Language-fallback pass done. Recovered {n_recovered}/{len(no_python)} "
          f"({n_still_none} genuinely had no code in any language).")
    print("Final status counts:", Counter(d["status"] for d in cache.values()))


def main(limit=None):
    fetch_python_editorials(limit=limit)
    fetch_language_fallback(limit=limit)
    cache = load(CACHE_PATH, {})
    print(f"\nTotal cached: {len(cache)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
