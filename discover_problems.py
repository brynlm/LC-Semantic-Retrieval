"""
Discovers the full current LeetCode algorithms-category problem list via the
live `questionList` GraphQL query (paginated over skip/limit). This is the
single source of truth for "what problems exist" for the whole pipeline --
no frozen dataset snapshot involved anywhere. "New" here means "appeared on
LeetCode since the last time this ran," computed by diffing against
whatever this same script last wrote, not against any external dataset.

Pure discovery/metadata pass -- title, slug, difficulty, tags, paid-only
status. Does NOT fetch descriptions or solution code (see
fetch_problem_metadata.py and fetch_problem_editorials.py for those);
keeping this step narrow means re-running it to pick up brand-new problems
later is cheap and doesn't reshuffle anything downstream.

Writes problem_catalog_cache.pkl: {titleSlug: {questionFrontendId, title,
difficulty, isPaidOnly, tags}} for every problem LeetCode returns -- the
full listing, not just the newly-appeared subset, so every downstream stage
can compute its own "pending" set directly against this one file.

Usage:
    python discover_problems.py
"""

import os
import pickle
import time

import requests
from dotenv import load_dotenv

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
CACHE_PATH = "problem_catalog_cache.pkl"


def load(path, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def fetch_page(skip: int, limit: int):
    # Scoped to "algorithms" specifically -- LeetCode's categorySlug values
    # split cleanly: algorithms/database/pandas/shell/concurrency/javascript
    # are separate pools, and this project's whole pipeline (abstracts,
    # embeddings) is built around general DSA/algorithm solutions -- SQL and
    # pandas-DataFrame problems don't fit that model even though some
    # superficially have a python-shaped signature.
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
    previous = load(CACHE_PATH, {})
    print(f"Catalog from last run: {len(previous)} problems")

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

    save(discovered, CACHE_PATH)

    new_slugs = sorted(set(discovered) - set(previous))
    new_free = [n for n in new_slugs if not discovered[n]["isPaidOnly"]]
    new_paid = [n for n in new_slugs if discovered[n]["isPaidOnly"]]
    print(f"\nTotal on LeetCode: {len(discovered)}")
    print(f"New since last discovery run: {len(new_slugs)} ({len(new_free)} free, {len(new_paid)} paid-only)")
    print(f"Saved full listing to {CACHE_PATH}")


if __name__ == "__main__":
    main()
