"""
Finishes scraping community "Solutions" tab posts across the full dataset,
this time actually looking for a VERIFIED Python solution per problem rather
than just caching whatever the top-voted post happens to contain (that was
v1's approach, and we found ~49% of top posts aren't Python at all). For
each problem, walks its candidates in vote order, fetches content, and runs
it through community_solution_extract.find_verified_solution (mistune-based
parsing + ast.parse + execution against the real test harness) until one
verifies or the candidate budget is exhausted.

Reuses prior work rather than re-fetching from scratch:
  - community_solution_status_cache.pkl (v1): premium_locked names skip the
    is_premium_only check entirely.
  - community_solution_candidates_cache.pkl (v1): candidate lists already
    fetched for ~900 problems are reused as-is.
  - community_solution_content_cache.pkl (v1): the one candidate's content
    already fetched per problem is reused as the first thing tried, instead
    of re-fetching it.

New, separate output caches (production caches are untouched):
  - community_verified_status_cache.pkl  -- name -> status enum
  - community_verified_solution_cache.pkl -- name -> verified python code
  - community_verified_meta_cache.pkl     -- name -> winning post's metadata
    (topicId, title, author, votes) for provenance/attribution

Resumable: skips any name already in community_verified_status_cache.pkl.

Usage:
    python run_community_solution_scrape_v2.py
"""

import os
import pickle
import time
from collections import Counter

import requests

import categorical_similarity as cs
from community_solution_extract import find_verified_solution

GRAPHQL_URL = "https://leetcode.com/graphql"
REQUEST_DELAY_S = 0.3
MAX_CANDIDATES_PER_PROBLEM = 6

V1_STATUS_PATH = "community_solution_status_cache.pkl"
V1_CANDIDATES_PATH = "community_solution_candidates_cache.pkl"
V1_CONTENT_PATH = "community_solution_content_cache.pkl"

STATUS_PATH = "community_verified_status_cache.pkl"
SOLUTION_PATH = "community_verified_solution_cache.pkl"
META_PATH = "community_verified_meta_cache.pkl"

STATUS_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) { isPaidOnly }
}
"""

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


def is_premium_only(name: str) -> bool:
    r = requests.post(GRAPHQL_URL, json={"query": STATUS_QUERY, "variables": {"titleSlug": name}}, timeout=15)
    data = r.json().get("data", {}).get("question")
    return bool(data and data.get("isPaidOnly"))


def fetch_candidates(name: str, top_n: int = 8) -> list[dict]:
    r = requests.post(
        GRAPHQL_URL,
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
    r = requests.post(GRAPHQL_URL, json={"query": CONTENT_QUERY, "variables": {"topicId": topic_id}}, timeout=15)
    node = r.json().get("data", {}).get("ugcArticleSolutionArticle")
    content = node.get("content") if node else None
    return content or None


def main():
    df = cs.load_dataset()
    names = sorted(df["name"].unique())

    v1_status = load(V1_STATUS_PATH, {})
    v1_candidates = load(V1_CANDIDATES_PATH, {})
    v1_content = load(V1_CONTENT_PATH, {})

    status = load(STATUS_PATH, {})
    solutions = load(SOLUTION_PATH, {})
    meta = load(META_PATH, {})

    pending = [n for n in names if n not in status]
    print(f"Total: {len(names)}, already processed: {len(names) - len(pending)}, pending: {len(pending)}")

    for i, name in enumerate(pending):
        row = df[df["name"] == name].iloc[0]
        try:
            if v1_status.get(name) == "premium_locked" or (name not in v1_status and is_premium_only(name)):
                status[name] = "premium_locked"
            else:
                candidates = v1_candidates.get(name)
                if candidates is None:
                    candidates = fetch_candidates(name)
                    time.sleep(REQUEST_DELAY_S)

                if not candidates:
                    status[name] = "no_candidates"
                else:
                    found = False
                    v1_pick = v1_content.get(name)  # {"topicId":, "content":, ...} already fetched in v1
                    for c in candidates[:MAX_CANDIDATES_PER_PROBLEM]:
                        if v1_pick and v1_pick["topicId"] == c["topicId"]:
                            content = v1_pick["content"]
                        else:
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

        if (i + 1) % 50 == 0:
            save(status, STATUS_PATH)
            save(solutions, SOLUTION_PATH)
            save(meta, META_PATH)
            print(f"  ...{i + 1}/{len(pending)} processed ({Counter(status.values())})")

    save(status, STATUS_PATH)
    save(solutions, SOLUTION_PATH)
    save(meta, META_PATH)
    print(f"\nDone. Final status breakdown: {Counter(status.values())}")


if __name__ == "__main__":
    main()
