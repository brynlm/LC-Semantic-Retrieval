"""
Scrapes community "Solutions" tab posts as a candidate source of optimal-ish
code for problems with no official editorial. Confirmed empirically: for any
problem where isPaidOnly=False, the entire Solutions tab (list + full post
content) is public -- no session/cookie needed at all. Only LeetCode's own
Premium-exclusive *problems* (isPaidOnly=True) require a valid session for
anything about them, editorial or community.

Stores raw scrape output ONLY -- no code extraction, no abstract generation,
no merging into production caches. That's deliberately staged as separate
follow-up steps so each part can be reviewed independently:
  1. community_solution_status_cache.pkl   -- name -> status enum
  2. community_solution_candidates_cache.pkl -- name -> [{uuid, title, topicId,
     author, votes} ...] (top N by vote count, metadata only)
  3. community_solution_content_cache.pkl  -- name -> the first candidate (in
     vote order) whose full content fetch actually succeeded, raw markdown
     content included verbatim (not yet parsed for a python code block)

Resumable: skips any name already in the status cache. Progress saved
incrementally.

Usage:
    python run_community_solution_scrape.py
"""

import os
import pickle
import time

import requests

import categorical_similarity as cs

GRAPHQL_URL = "https://leetcode.com/graphql"
REQUEST_DELAY_S = 0.3

STATUS_CACHE_PATH = "community_solution_status_cache.pkl"
CANDIDATES_CACHE_PATH = "community_solution_candidates_cache.pkl"
CONTENT_CACHE_PATH = "community_solution_content_cache.pkl"

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
    status = load(STATUS_CACHE_PATH, {})
    candidates_cache = load(CANDIDATES_CACHE_PATH, {})
    content_cache = load(CONTENT_CACHE_PATH, {})

    names = sorted(df["name"].unique())
    pending = [n for n in names if n not in status]
    print(f"Total problems: {len(names)}, already processed: {len(names) - len(pending)}, pending: {len(pending)}")

    done_count = 0
    for i, name in enumerate(pending):
        try:
            if is_premium_only(name):
                status[name] = "premium_locked"
            else:
                cands = fetch_candidates(name)
                candidates_cache[name] = cands
                if not cands:
                    status[name] = "no_candidates"
                else:
                    picked = None
                    for c in cands[:5]:
                        time.sleep(REQUEST_DELAY_S)
                        content = fetch_content(c["topicId"])
                        if content:
                            picked = {**c, "content": content}
                            break
                    if picked:
                        content_cache[name] = picked
                        status[name] = "done"
                        done_count += 1
                    else:
                        status[name] = "content_fetch_failed"
        except Exception as e:
            status[name] = f"error: {str(e)[:100]}"

        time.sleep(REQUEST_DELAY_S)

        if (i + 1) % 50 == 0:
            save(status, STATUS_CACHE_PATH)
            save(candidates_cache, CANDIDATES_CACHE_PATH)
            save(content_cache, CONTENT_CACHE_PATH)
            from collections import Counter
            print(f"  ...{i + 1}/{len(pending)} processed ({Counter(status.values())})")

    save(status, STATUS_CACHE_PATH)
    save(candidates_cache, CANDIDATES_CACHE_PATH)
    save(content_cache, CONTENT_CACHE_PATH)
    from collections import Counter
    print(f"\nDone. {done_count} content fetches succeeded. Final status breakdown: {Counter(status.values())}")


if __name__ == "__main__":
    main()
