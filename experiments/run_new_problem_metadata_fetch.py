"""
Fetches description/starter-code/example-testcases for every FREE problem
discovered by run_problem_discovery.py that isn't already in our corpus.
Paid-only new problems (435 of the 1414 discovered) are skipped here --
their content isn't accessible without a LeetCode premium account
regardless of session cookie, so they're left for a separate pass if that
ever becomes available; run_problem_discovery.py's cache already flags
them via isPaidOnly so nothing about them is lost.

Deliberately narrow scope, matching the run_full_pool_expansion.py-style
split elsewhere in this project: this script only gets what a problem
*is* (description, starter code, entry point, tags/difficulty already
captured by discovery, raw example testcases for future test-harness
work). Solution code is a separate step (run_new_problem_editorial_fetch.py)
so that re-running one doesn't force re-fetching the other.

Does NOT synthesize an executable test harness from exampleTestcases --
that's deferred (see PROJECT_CONTEXT discussion: LeetCode's raw examples
are input-only text blocks, not assert-ready pairs, and building a general
parser for that is real, separate work). exampleTestcases/metaData are
still captured now since they're free to grab alongside content, so
that future work doesn't need to re-scrape.

Writes new_problem_metadata_cache.pkl: {slug: {description, starter_code,
entry_point, example_testcases, difficulty, tags, is_paid_only}}

Usage:
    python run_new_problem_metadata_fetch.py [--limit N]
"""

import argparse
import json
import os
import pickle
import time

import requests
from dotenv import load_dotenv

import categorical_similarity as cs

load_dotenv()

_SESSION = os.environ.get("LEETCODE_SESSION")
_CSRF = os.environ.get("LEETCODE_CSRF_TOKEN")
_COOKIES = {"LEETCODE_SESSION": _SESSION, "csrftoken": _CSRF} if _SESSION else {}
_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "x-csrftoken": _CSRF or "",
    "Referer": "https://leetcode.com/",
    "Content-Type": "application/json",
}

_QUERY = """
query questionContent($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    content
    isPaidOnly
    codeSnippets { langSlug code }
    exampleTestcases
    metaData
  }
}
"""

DISCOVERED_PATH = "discovered_problems_cache.pkl"
CACHE_PATH = "new_problem_metadata_cache.pkl"
CHECKPOINT_EVERY = 20
REQUEST_DELAY_S = 0.4
MAX_RETRIES = 3


def load(path, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def fetch_one(slug):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                "https://leetcode.com/graphql",
                json={"query": _QUERY, "variables": {"titleSlug": slug}},
                cookies=_COOKIES, headers=_HEADERS, timeout=15,
            )
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()["data"]["question"]
            if data is None:
                return None  # slug not found / inaccessible
            return data
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"  error on {slug} (attempt {attempt + 1}): {e}")
            time.sleep(2 * (attempt + 1))
    return "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process the first N new free slugs (for testing)")
    args = ap.parse_args()

    discovered = load(DISCOVERED_PATH, {})
    known_names = set(cs.load_dataset()["name"])
    new_free = sorted(n for n in discovered if n not in known_names and not discovered[n]["isPaidOnly"])
    print(f"New free problems to fetch: {len(new_free)}")

    cache = load(CACHE_PATH, {})
    pending = [n for n in new_free if n not in cache]
    if args.limit:
        pending = pending[:args.limit]
    print(f"Already cached: {len(new_free) - len(pending) if not args.limit else 'n/a (limited run)'}; processing {len(pending)} now")

    n_ok, n_not_python, n_failed = 0, 0, 0
    for i, slug in enumerate(pending):
        data = fetch_one(slug)
        if data is None:
            cache[slug] = {"status": "not_found"}
            n_failed += 1
        elif data == "error":
            cache[slug] = {"status": "error"}
            n_failed += 1
        else:
            python_snippet = next((c["code"] for c in data["codeSnippets"] if c["langSlug"] == "python3"), None)
            try:
                meta = json.loads(data["metaData"])
                entry_point = meta.get("name")
            except (ValueError, TypeError):
                entry_point = None
            # No python3 codeSnippet at all means this isn't a "solution
            # missing" gap to fill later -- it's a category mismatch (SQL
            # problems tagged "Database", or JS-only problems built around
            # Promise/setTimeout with no Python equivalent). Confirmed on a
            # sample: these have no starter code in ANY Python variant, not
            # just an unfetched one. Keep them distinct from "ok" so the
            # abstract-generation step never tries to process them.
            status = "ok" if python_snippet else "not_python"
            cache[slug] = {
                "status": status,
                "description": data["content"],
                "starter_code": python_snippet,
                "entry_point": entry_point,
                "example_testcases": data["exampleTestcases"],
                "difficulty": discovered[slug]["difficulty"],
                "tags": discovered[slug]["tags"],
                "is_paid_only": data["isPaidOnly"],
            }
            if status == "ok":
                n_ok += 1
            else:
                n_not_python += 1
        time.sleep(REQUEST_DELAY_S)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save(cache, CACHE_PATH)
            print(f"  ...{i + 1}/{len(pending)} processed (ok={n_ok}, not_python={n_not_python}, failed={n_failed})")

    save(cache, CACHE_PATH)
    print(f"\nDone. {n_ok} python-eligible, {n_not_python} not_python (SQL/JS-only/etc.), "
          f"{n_failed} failed, {len(cache)} total cached.")


if __name__ == "__main__":
    main()
