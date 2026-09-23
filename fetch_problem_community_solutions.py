"""
Fetches community "Solutions" tab candidates for problems with no usable
editorial (fetch_problem_editorials.py status "no_editorial" or
"editorial_no_python" with no code in any language). Reuses
community_solution_extract's fetch_candidates/fetch_content (slug-only, no
dataset dependency) and its mistune-based extract_code_blocks_ranked
unchanged.

Deliberately does NOT attempt execution-based verification -- that needs a
real test harness (prompt/test/entry_point/assert pairs), which LeetCode's
raw exampleTestcases field doesn't provide ready-made (building a
synthesizer for that is real, separate work, deferred; see
fetch_problem_metadata.py's docstring). So this is a lower-rigor pass by
design, uniformly, for every problem in the corpus: pick the best-ranked
code block from the top-voted posts and sanity-check it, rather than
proving it correct by running it. Correctness was never a claim this
pipeline makes anyway -- the abstract-generation step describes a
solution's mechanism, not its certified correctness.

Two-tier acceptance, in order:
  1. A Python-tagged code block (from extract_code_blocks_ranked's
     py_tagged list) that parses and contains a function/method def ->
     status "community_unverified", language "python". Walks candidates in
     vote order until one yields such a block.
  2. If no candidate yields a valid Python block, fall back to the
     top-voted candidate's first code block of ANY language, untagged and
     unchecked -> status "community_unverified_nonpython".
  3. No code blocks found in any candidate -> status "no_community_solution".

Writes problem_community_cache.pkl: {slug: {status, code, language,
meta: {topicId, title, author, votes}}}.

Usage:
    python fetch_problem_community_solutions.py [--limit N]
"""

import argparse
import pickle
import textwrap
import time
from collections import Counter

from community_solution_extract import extract_code_blocks_ranked, fetch_candidates, fetch_content, normalize_escapes
from pipeline_lib import is_usable_solution_code

EDITORIAL_CACHE_PATH = "problem_editorial_cache.pkl"
CACHE_PATH = "problem_community_cache.pkl"
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


def _as_usable_python(code: str):
    """Returns a usable, syntactically-complete version of `code` if one
    exists, else None. Tries the block as-is first, then wrapped in a
    synthetic `class Solution:` with the whole block re-indented one level
    -- some community posts' code fences only include the methods, not the
    enclosing class declaration. Deliberately does NOT attempt to repair
    genuine syntax errors -- guessing at a fix risks silently shipping wrong
    code, which is worse than leaving it correctly flagged as unusable."""
    if is_usable_solution_code(code):
        return code
    wrapped = "class Solution:\n" + textwrap.indent(code, "    ")
    if is_usable_solution_code(wrapped):
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


def main(limit=None):
    editorial_cache = load(EDITORIAL_CACHE_PATH, {})
    needs_community = sorted(
        s for s, d in editorial_cache.items()
        if d["status"] in ("no_editorial", "editorial_no_python")
    )
    print(f"Problems needing community fallback: {len(needs_community)}")

    cache = load(CACHE_PATH, {})
    pending = [n for n in needs_community if n not in cache]
    if limit:
        pending = pending[:limit]
    print(f"Processing {len(pending)} now")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
