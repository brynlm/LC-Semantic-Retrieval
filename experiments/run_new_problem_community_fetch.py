"""
Fetches community "Solutions" tab candidates for the 554 new problems with
no usable editorial (new_problem_editorial_cache.pkl status "no_editorial"
or "editorial_no_python" with no code in any language). Reuses
run_community_solution_scrape_v2.py's fetch_candidates/fetch_content
(slug-only, no HF dependency) and community_solution_extract's mistune-based
extract_code_blocks_ranked unchanged.

Deliberately does NOT attempt execution-based verification the way
run_community_solution_scrape_v2.py's find_verified_solution does -- that
needs prompt/test/entry_point/starter_code, which only exist as HF dataset
columns and don't exist for problems outside that snapshot (confirmed this
session; building a test-harness synthesizer from LeetCode's raw
exampleTestcases is real, separate work, deferred). So this is a lower-
rigor pass: pick the best-ranked code block from the top-voted posts and
sanity-check it, rather than proving it correct by running it.

Two-tier acceptance, in order, matching the project's language-agnostic
stance for abstract generation (the LLM describes the mechanism regardless
of source language -- only the sanity check itself is Python-specific,
since structural_features.is_usable_solution_code() needs ast.parse):
  1. A Python-tagged code block (from extract_code_blocks_ranked's
     py_tagged list) that parses and contains a function/method def ->
     status "community_unverified", language "python". Walks candidates in
     vote order until one yields such a block.
  2. If no candidate yields a valid Python block, fall back to the
     top-voted candidate's first code block of ANY language, untagged and
     unchecked (we can't ast.parse non-Python) -> status
     "community_unverified_nonpython". Lower confidence still, but kept
     and flagged rather than discarded, per the "flag gaps, don't skip
     them" approach.
  3. No code blocks found in any candidate -> status "no_community_solution".

Writes new_problem_community_cache.pkl: {slug: {status, code, language,
meta: {topicId, title, author, votes}}}. Does not touch any production
cache or the v1/v2 community caches for the existing corpus.

Usage:
    python run_new_problem_community_fetch.py [--limit N]
"""

import argparse
import pickle
import textwrap
import time

import structural_features as sf
from community_solution_extract import extract_code_blocks_ranked, normalize_escapes
from run_community_solution_scrape_v2 import fetch_candidates, fetch_content

EDITORIAL_CACHE_PATH = "new_problem_editorial_cache.pkl"
CACHE_PATH = "new_problem_community_cache.pkl"
CHECKPOINT_EVERY = 15
REQUEST_DELAY_S = 0.3
MAX_CANDIDATES_PER_PROBLEM = 6


def load(path, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _as_usable_python(code: str) -> str | None:
    """Returns a usable, syntactically-complete version of `code` if one
    exists, else None. Tries the block as-is first, then wrapped in a
    synthetic `class Solution:` with the whole block re-indented one level
    -- confirmed on a real sample that ~38% of blocks initially classified
    "not Python" were actually valid Python method bodies whose community
    post's code fence only included the methods, not the enclosing class
    declaration (the class line was written in separate prose or a
    different fence), which is a bare indented `def` at the top level and
    therefore a guaranteed SyntaxError on its own even though the code
    itself is fine. Deliberately does NOT attempt to repair genuine syntax
    errors (e.g. a missing closing paren) -- confirmed those exist too in a
    handful of posts, and guessing at a fix risks silently shipping wrong
    code, which is a different and much worse failure than just leaving it
    correctly flagged as unusable."""
    if sf.is_usable_solution_code(code):
        return code
    wrapped = "class Solution:\n" + textwrap.indent(code, "    ")
    if sf.is_usable_solution_code(wrapped):
        return wrapped
    return None


def process_one(slug):
    candidates = fetch_candidates(slug, top_n=8)
    if not candidates:
        return {"status": "no_community_solution", "code": None, "language": None, "meta": None}

    candidates = sorted(candidates, key=lambda c: -c["votes"])[:MAX_CANDIDATES_PER_PROBLEM]

    first_fallback = None
    for cand in candidates:
        content = fetch_content(cand["topicId"])
        time.sleep(REQUEST_DELAY_S)
        if not content:
            continue
        blocks = extract_code_blocks_ranked(content)
        if not blocks:
            continue
        if first_fallback is None:
            first_fallback = (normalize_escapes(blocks[0]), cand)
        for block in blocks:
            usable = _as_usable_python(normalize_escapes(block))
            if usable:
                return {"status": "community_unverified", "code": usable, "language": "python", "meta": cand}

    if first_fallback:
        code, cand = first_fallback
        return {"status": "community_unverified_nonpython", "code": code, "language": "unknown", "meta": cand}
    return {"status": "no_community_solution", "code": None, "language": None, "meta": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    editorial_cache = load(EDITORIAL_CACHE_PATH, {})
    needs_community = sorted(
        s for s, d in editorial_cache.items()
        if d["status"] in ("no_editorial", "editorial_no_python")
    )
    print(f"Problems needing community fallback: {len(needs_community)}")

    cache = load(CACHE_PATH, {})
    pending = [n for n in needs_community if n not in cache]
    if args.limit:
        pending = pending[:args.limit]
    print(f"Processing {len(pending)} now")

    from collections import Counter
    counts = Counter()
    for i, slug in enumerate(pending):
        try:
            cache[slug] = process_one(slug)
        except Exception as e:
            cache[slug] = {"status": f"error: {type(e).__name__}: {str(e)[:100]}", "code": None, "language": None, "meta": None}
        counts[cache[slug]["status"].split(":")[0]] += 1
        time.sleep(REQUEST_DELAY_S)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save(cache, CACHE_PATH)
            print(f"  ...{i + 1}/{len(pending)}  {dict(counts)}")

    save(cache, CACHE_PATH)
    print(f"\nDone. Final counts this run: {dict(counts)}")
    print(f"Total cached: {len(cache)}")


if __name__ == "__main__":
    main()
