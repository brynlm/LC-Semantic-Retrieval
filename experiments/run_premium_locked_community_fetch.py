"""
Re-attempts the community-solution fetch+verify pass for problems that were
skipped as premium_locked in run_community_solution_scrape_v2.py, now using
an authenticated session (needed since viewing full post content on a
Premium-only problem requires it, even though the same fetch is public for
free problems).

Usage:
    python run_premium_locked_community_fetch.py
"""

import os
import pickle
import time
from collections import Counter

import requests
from dotenv import load_dotenv

load_dotenv()

import categorical_similarity as cs
from community_solution_extract import find_verified_solution

GRAPHQL_URL = "https://leetcode.com/graphql"
REQUEST_DELAY_S = 0.3
MAX_CANDIDATES_PER_PROBLEM = 6

STATUS_PATH = "community_verified_status_cache.pkl"
SOLUTION_PATH = "community_verified_solution_cache.pkl"
META_PATH = "community_verified_meta_cache.pkl"

_SESSION = os.environ.get("LEETCODE_SESSION")
_CSRF = os.environ.get("LEETCODE_CSRF_TOKEN")
_COOKIES = {"LEETCODE_SESSION": _SESSION, "csrftoken": _CSRF}
_HEADERS = {"x-csrftoken": _CSRF or "", "Content-Type": "application/json", "Referer": "https://leetcode.com"}

LIST_QUERY = """
query communitySolutions($questionSlug: String!, $skip: Int!, $first: Int!) {
  ugcArticleSolutionArticles(questionSlug: $questionSlug, skip: $skip, first: $first, orderBy: MOST_VOTES) {
    edges {
      node {
        uuid
        title
        topicId
        author { userName }
        reactions { count reactionType }
      }
    }
  }
}
"""

CONTENT_QUERY = """
query solutionArticle($topicId: ID!) {
  ugcArticleSolutionArticle(topicId: $topicId) { content }
}
"""


def load(path, default):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def fetch_candidates(name: str, top_n: int = 8) -> list[dict]:
    r = requests.post(
        GRAPHQL_URL, cookies=_COOKIES, headers=_HEADERS,
        json={"query": LIST_QUERY, "variables": {"questionSlug": name, "skip": 0, "first": top_n}},
        timeout=15,
    )
    edges = r.json().get("data", {}).get("ugcArticleSolutionArticles", {}).get("edges", []) or []
    out = []
    for e in edges:
        n = e["node"]
        votes = next((rx["count"] for rx in (n.get("reactions") or []) if rx["reactionType"] == "UPVOTE"), 0)
        out.append({
            "uuid": n["uuid"], "title": n["title"], "topicId": n["topicId"],
            "author": n["author"]["userName"], "votes": votes,
        })
    return out


def fetch_content(topic_id: int) -> str | None:
    r = requests.post(
        GRAPHQL_URL, cookies=_COOKIES, headers=_HEADERS,
        json={"query": CONTENT_QUERY, "variables": {"topicId": topic_id}}, timeout=15,
    )
    node = r.json().get("data", {}).get("ugcArticleSolutionArticle")
    content = node.get("content") if node else None
    return content or None


def main():
    df = cs.load_dataset()
    status = load(STATUS_PATH, {})
    solutions = load(SOLUTION_PATH, {})
    meta = load(META_PATH, {})

    premium_locked = [n for n, s in status.items() if s == "premium_locked"]
    print(f"Premium-locked problems to retry with auth: {len(premium_locked)}")

    for i, name in enumerate(premium_locked):
        row = df[df["name"] == name].iloc[0]
        try:
            candidates = fetch_candidates(name)
            time.sleep(REQUEST_DELAY_S)
            if not candidates:
                status[name] = "no_candidates"
            else:
                found = False
                for c in candidates[:MAX_CANDIDATES_PER_PROBLEM]:
                    content = fetch_content(c["topicId"])
                    time.sleep(REQUEST_DELAY_S)
                    if not content:
                        continue
                    code = find_verified_solution(
                        content, row["prompt"], row["test"], row["entry_point"], row["starter_code"]
                    )
                    if code:
                        solutions[name] = code
                        meta[name] = c
                        status[name] = "verified"
                        found = True
                        break
                if not found:
                    status[name] = "no_verified_python_found"
        except Exception as e:
            status[name] = f"error: {str(e)[:100]}"

        if (i + 1) % 25 == 0:
            save(status, STATUS_PATH)
            save(solutions, SOLUTION_PATH)
            save(meta, META_PATH)
            print(f"  ...{i + 1}/{len(premium_locked)} processed")

    save(status, STATUS_PATH)
    save(solutions, SOLUTION_PATH)
    save(meta, META_PATH)
    remaining = Counter(status[n] for n in premium_locked)
    print(f"\nDone. Outcomes for the previously-premium-locked set: {remaining}")


if __name__ == "__main__":
    main()
