"""
Discovers the full current LeetCode problem list via the live
`questionList` GraphQL query (paginated over skip/limit) and diffs it
against our existing corpus (the frozen HF LeetCodeDataset snapshot) to
find problems we don't have at all -- e.g. anything added to LeetCode after
that snapshot was created. Confirmed live: LeetCode currently lists 4055
problems total vs. our ~2641-problem corpus, so this is a substantial,
real expansion, not a handful of edge cases.

This is a pure discovery/metadata pass -- title, slug, difficulty, tags,
paid-only status. It does NOT fetch descriptions or solution code (see
run_problem_metadata_fetch.py and run_new_problem_editorial_fetch.py for
those); keeping this step narrow means re-running it to pick up brand-new
LeetCode problems later is cheap and doesn't reshuffle anything downstream.

Writes discovered_problems_cache.pkl: {titleSlug: {questionFrontendId,
title, difficulty, isPaidOnly, tags}} for every problem LeetCode returns,
known or not -- keeping the full listing (not just the new-to-us subset)
means future diffs against a growing corpus don't require re-fetching.

Usage:
    python run_problem_discovery.py
"""

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
query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
  problemsetQuestionList: questionList(
    categorySlug: $categorySlug
    limit: $limit
    skip: $skip
    filters: $filters
  ) {
    total: totalNum
    questions: data {
      questionFrontendId
      title
      titleSlug
      difficulty
      isPaidOnly
      topicTags { name }
    }
  }
}
"""

PAGE_SIZE = 100
REQUEST_DELAY_S = 0.4
MAX_RETRIES = 3
CACHE_PATH = "discovered_problems_cache.pkl"


def fetch_page(skip: int, limit: int):
    # Scoped to "algorithms" specifically -- confirmed via direct query that
    # LeetCode's categorySlug values split cleanly: algorithms=3637,
    # database=323, pandas=338 (overlaps database -- same problems, a
    # pandas-solution track), shell=4, concurrency=9, javascript=67. This
    # project's whole pipeline (abstracts, code-only embeddings) is built
    # around general DSA/algorithm solutions -- SQL and pandas-DataFrame
    # problems don't fit that model even though some superficially have a
    # python-shaped signature (confirmed: a plain "does python3 code exist"
    # check does NOT catch pandas problems, since their solutions are valid
    # Python, just not algorithmic). Filtering at the category level here is
    # more reliable than any post-hoc content check.
    variables = {"categorySlug": "algorithms", "skip": skip, "limit": limit, "filters": {}}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                "https://leetcode.com/graphql",
                json={"query": _QUERY, "variables": variables},
                cookies=_COOKIES, headers=_HEADERS, timeout=15,
            )
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()["data"]["problemsetQuestionList"]
            return data["total"], data["questions"]
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"  error at skip={skip} (attempt {attempt + 1}): {e}")
            time.sleep(2 * (attempt + 1))
    return None, None


def main():
    known_names = set(cs.load_dataset()["name"])
    print(f"Existing corpus (HF snapshot): {len(known_names)} problems")

    discovered = {}
    skip = 0
    total = None
    while total is None or skip < total:
        t, questions = fetch_page(skip, PAGE_SIZE)
        if questions is None:
            print(f"Giving up at skip={skip} after repeated failures.")
            break
        total = t
        for q in questions:
            discovered[q["titleSlug"]] = {
                "questionFrontendId": q["questionFrontendId"],
                "title": q["title"],
                "difficulty": q["difficulty"],
                "isPaidOnly": q["isPaidOnly"],
                "tags": [t["name"] for t in q["topicTags"]],
            }
        skip += PAGE_SIZE
        print(f"  ...{min(skip, total)}/{total} fetched")
        time.sleep(REQUEST_DELAY_S)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(discovered, f)

    new_slugs = sorted(set(discovered) - known_names)
    new_free = [n for n in new_slugs if not discovered[n]["isPaidOnly"]]
    new_paid = [n for n in new_slugs if discovered[n]["isPaidOnly"]]
    print(f"\nTotal discovered on LeetCode: {len(discovered)}")
    print(f"New to our corpus: {len(new_slugs)} ({len(new_free)} free, {len(new_paid)} paid-only)")
    print(f"Saved full listing to {CACHE_PATH}")


if __name__ == "__main__":
    main()
