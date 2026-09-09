import os
import re
import pickle

import requests
from dotenv import load_dotenv

load_dotenv()

session = os.environ["LEETCODE_SESSION"]
csrf = os.environ["LEETCODE_CSRF_TOKEN"]
COOKIES = {"LEETCODE_SESSION": session, "csrftoken": csrf}
HEADERS = {"User-Agent": "Mozilla/5.0", "x-csrftoken": csrf, "Referer": "https://leetcode.com/", "Content-Type": "application/json"}

CONTENT_QUERY = """query officialSolution($titleSlug: String!) { ugcArticleOfficialSolutionArticle(questionSlug: $titleSlug) { uuid title content } }"""
PLAYGROUND_QUERY = """query playgroundData($uuid: String!) { allPlaygroundCodes(uuid: $uuid) { code langSlug } }"""


def fetch_official_python_solution(name: str) -> str | None:
    resp = requests.post("https://leetcode.com/graphql", json={"query": CONTENT_QUERY, "variables": {"titleSlug": name}}, cookies=COOKIES, headers=HEADERS, timeout=15)
    article = resp.json().get("data", {}).get("ugcArticleOfficialSolutionArticle")
    if not article:
        return None
    content = article["content"]
    uuids = re.findall(r"playground/([A-Za-z0-9]+)/shared", content)
    if not uuids:
        return None
    # try from last (most optimized) backwards until we find a python3 variant
    for uuid in reversed(uuids):
        pg_resp = requests.post("https://leetcode.com/graphql", json={"query": PLAYGROUND_QUERY, "variables": {"uuid": uuid}}, cookies=COOKIES, headers=HEADERS, timeout=15)
        codes = pg_resp.json().get("data", {}).get("allPlaygroundCodes") or []
        for c in codes:
            if c["langSlug"] == "python3":
                return c["code"]
    return None


NAMES = ["sliding-window-maximum", "jump-game-vi", "course-schedule-ii", "alien-dictionary", "minimum-height-trees"]

results = {}
for name in NAMES:
    code = fetch_official_python_solution(name)
    results[name] = code
    print(f"=== {name} ===")
    print(code if code else "(no python3 solution found)")
    print()

with open("/tmp/official_editorial_code.pkl", "wb") as f:
    pickle.dump(results, f)
