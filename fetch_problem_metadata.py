"""
Fetches description/starter-code/example-testcases for every FREE problem in
problem_catalog_cache.pkl not already in problem_metadata_cache.pkl. Paid-only
problems are skipped -- their content isn't accessible without a LeetCode
premium account regardless of session cookie; problem_catalog_cache.pkl
already flags them via isPaidOnly so nothing about them is lost.

Deliberately narrow scope: this script only gets what a problem *is*
(description, starter code, entry point, tags/difficulty already captured by
discovery, raw example testcases for future test-harness work). Solution
code is a separate step (fetch_problem_editorials.py) so that re-running one
doesn't force re-fetching the other.

Does NOT synthesize an executable test harness from exampleTestcases --
LeetCode's raw examples are input-only text blocks, not assert-ready pairs,
and building a general parser for that is real, separate work (deferred).
exampleTestcases/metaData are still captured now since they're free to grab
alongside content, so that future work doesn't need to re-scrape.

`description` is converted from LeetCode's raw HTML (its `content` field, the
same markup leetcode.com itself renders) to plain text before being cached
-- the static site's client displays it via escapeHtml(), i.e. it expects
plain text and would otherwise show literal "<p>"/"<img>" tags to visitors.

Writes problem_metadata_cache.pkl: {slug: {status, description, starter_code,
entry_point, example_testcases, difficulty, tags, is_paid_only}}. `status`
is "ok" (has a python3 starter) or "not_python" (SQL/pandas/JS-only problems
that slipped through the categorySlug filter with no Python solution
possible at all -- kept distinct from a fetch failure so this step never
tries to process them).

Usage:
    python fetch_problem_metadata.py [--limit N]
"""

import argparse
import html
import json
import os
import pickle
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()

_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_BLOCK_BREAK_RE = re.compile(r"</?(p|div|pre|ul|ol|li|br|h[1-6])\b[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_EXTRA_BLANKLINES_RE = re.compile(r"\n{3,}")


def html_to_text(content: str) -> str:
    """Strips LeetCode's problem-statement HTML down to plain text: drops
    <img> tags entirely (no alt-text fallback -- the original plain-text
    corpus this replaced didn't carry image references either), turns
    block-level tag boundaries into newlines so paragraphs/examples/lists
    stay visually separated, strips every remaining tag, and unescapes
    entities (e.g. &nbsp; -> the literal non-breaking-space character,
    matching how the site's existing text already represents it)."""
    if not content:
        return content
    text = _IMG_RE.sub("", content)
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _EXTRA_BLANKLINES_RE.sub("\n\n", text)
    return text.strip()

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

CATALOG_PATH = "problem_catalog_cache.pkl"
CACHE_PATH = "problem_metadata_cache.pkl"
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


def main(limit=None):
    catalog = load(CATALOG_PATH, {})
    free_slugs = sorted(n for n in catalog if not catalog[n]["isPaidOnly"])
    print(f"Free problems in catalog: {len(free_slugs)}")

    cache = load(CACHE_PATH, {})
    pending = [n for n in free_slugs if n not in cache]
    if limit:
        pending = pending[:limit]
    print(f"Already cached: {len(free_slugs) - len(pending) if not limit else 'n/a (limited run)'}; processing {len(pending)} now")

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
            # Promise/setTimeout with no Python equivalent). Keep distinct
            # from "ok" so the abstract-generation step never processes them.
            status = "ok" if python_snippet else "not_python"
            cache[slug] = {
                "status": status,
                "description": html_to_text(data["content"]),
                "starter_code": python_snippet,
                "entry_point": entry_point,
                "example_testcases": data["exampleTestcases"],
                "difficulty": catalog[slug]["difficulty"],
                "tags": catalog[slug]["tags"],
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process the first N pending slugs (for testing)")
    args = ap.parse_args()
    main(limit=args.limit)
