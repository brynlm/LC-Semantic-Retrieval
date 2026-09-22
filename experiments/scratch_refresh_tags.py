"""
Fetches fresh topicTags for every problem in the dataset from LeetCode's live
GraphQL API and diffs against our locally cached tags (from the static HF
LeetCodeDataset snapshot). Triggered by a concrete finding: "Minimax" is 0/2641
in our local tags column, even for Stone Game problems that carry it on the
live site -- meaning our local tag data has at least one real gap, not just
rare-but-present tags. Since tag-purity is now a load-bearing evaluation
signal (see scratch_cluster_purity.py), a systematic gap would quietly bias
it, so this checks the whole corpus rather than just the one confirmed case.

Does NOT overwrite lc_train_cache.pkl -- writes a diff report + a fresh tags
mapping to separate files for review before anything downstream is changed.

Resumable: checkpoints to /tmp/tag_refresh_checkpoint.pkl every 50 problems.

Usage:
    python scratch_refresh_tags.py
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

_TAGS_QUERY = """query questionTags($titleSlug: String!) { question(titleSlug: $titleSlug) { topicTags { name } } }"""

CHECKPOINT_PATH = "/tmp/tag_refresh_checkpoint.pkl"
CHECKPOINT_EVERY = 50
REQUEST_DELAY = 0.4
MAX_RETRIES = 3


def fetch_fresh_tags(name: str) -> list[str] | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://leetcode.com/graphql",
                json={"query": _TAGS_QUERY, "variables": {"titleSlug": name}},
                cookies=_COOKIES, headers=_HEADERS, timeout=15,
            )
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            data = resp.json().get("data", {}).get("question")
            if data is None:
                return None
            return [t["name"] for t in data["topicTags"]]
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    return None


def main():
    df = cs.load_dataset()
    names = df["name"].tolist()
    local_tags = dict(zip(df["name"], df["tags"]))

    fresh = {}
    not_found = []
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        fresh = ckpt["fresh"]
        not_found = ckpt["not_found"]
        print(f"Resuming from {len(fresh)} already-fetched, {len(not_found)} not-found")

    remaining = [n for n in names if n not in fresh and n not in not_found]
    print(f"Remaining to fetch: {remaining and len(remaining)}")

    for i, name in enumerate(remaining):
        tags = fetch_fresh_tags(name)
        if tags is None:
            not_found.append(name)
        else:
            fresh[name] = tags
        time.sleep(REQUEST_DELAY)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump({"fresh": fresh, "not_found": not_found}, f)
            print(f"  ...{i + 1}/{len(remaining)} fetched (ok={len(fresh)}, not_found={len(not_found)})")

    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({"fresh": fresh, "not_found": not_found}, f)

    # Diff against local
    diffs = []
    for name, fresh_tags in fresh.items():
        local_set = set(local_tags.get(name, []))
        fresh_set = set(fresh_tags)
        added = fresh_set - local_set   # LC has, we don't
        removed = local_set - fresh_set  # we have, LC doesn't
        if added or removed:
            diffs.append({"name": name, "local": sorted(local_set), "fresh": sorted(fresh_set),
                          "added": sorted(added), "removed": sorted(removed)})

    with open("/tmp/tag_refresh_diffs.pkl", "wb") as f:
        pickle.dump(diffs, f)
    with open("/tmp/tag_refresh_fresh_tags.pkl", "wb") as f:
        pickle.dump(fresh, f)

    print(f"\nDone. {len(fresh)} fetched, {len(not_found)} not found, {len(diffs)} problems with tag diffs.")
    if not_found:
        print(f"Not found (first 20): {not_found[:20]}")


if __name__ == "__main__":
    main()
