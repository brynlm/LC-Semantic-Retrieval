"""
Fetches official LeetCode editorial solutions via LeetCode's own (undocumented)
GraphQL API, authenticated with a personal Premium session so paywalled
editorials are included, not just the free subset.

Two-call pattern per problem:
  1. `ugcArticleOfficialSolutionArticle` -- the editorial's markdown content,
     which embeds one <iframe src=".../playground/<uuid>/shared"> per
     approach presented (not the code itself -- that page is
     Cloudflare-protected and not fetchable directly).
  2. `allPlaygroundCodes(uuid=...)` -- the actual code for one approach,
     across every language LeetCode offers; we filter for python3.

Requires LEETCODE_SESSION and LEETCODE_CSRF_TOKEN in the environment (from a
personal LeetCode Premium account's browser cookies) -- see .env.

Approach selection: editorials present multiple approaches per problem, and
their ordering convention is NOT consistent -- some go
worst-to-best (e.g. Two Sum: brute force -> two-pass hash -> one-pass hash,
last is best), others present the recommended approach first and slower
alternatives afterward for pedagogical completeness (e.g. Jump Game VI's
editorial explicitly says "we recommend Approach 1 ... since they have the
best performance", where Approach 1 is the FIRST one listed). Picking
"last approach" unconditionally silently reproduces the exact
solution-instance-diversity problem this scraper exists to fix. So approach
selection follows a priority chain:
  1. Parse an explicit "we recommend Approach N[, M, ...]" sentence (commonly
     found in an "Overview" section) and use the first recommended index.
  2. Otherwise, compare each approach's stated Big-O time complexity (parsed
     from "Time complexity: $$O(...)$$" lines) and pick the best-recognized
     one. Unrecognized complexity strings are skipped.
  3. Otherwise, fall back to the LAST approach (matches the more common
     worst-to-best convention empirically).

This is a heuristic, not a proof -- verify against known cases before trusting
it blindly at scale.

Usage:
    import leetcode_editorial_scraper as les
    code = les.fetch_official_python_solution("jump-game-vi")
"""

import os
import re

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

_CONTENT_QUERY = """query officialSolution($titleSlug: String!) { ugcArticleOfficialSolutionArticle(questionSlug: $titleSlug) { uuid title content } }"""
_PLAYGROUND_QUERY = """query playgroundData($uuid: String!) { allPlaygroundCodes(uuid: $uuid) { code langSlug } }"""
_STATUS_QUERY = """query questionData($titleSlug: String!) { question(titleSlug: $titleSlug) { isPaidOnly hasSolution solution { canSeeDetail paidOnly } } }"""

# Rough ordinal ranking of common Big-O complexity classes, for comparing
# approaches when no explicit recommendation is stated. Variable name in the
# expression (n, m, k, ...) is normalized away. Unrecognized forms map to None
# (excluded from comparison) rather than guessed at.
_COMPLEXITY_PATTERNS = [
    (r"O\(\s*1\s*\)", 0),
    (r"O\(\s*log\s*\(?\s*\w+\s*\)?\s*\)", 1),
    (r"O\(\s*\w+\s*\+\s*\w+\s*\)", 2),
    (r"O\(\s*\w+\s*\)", 2),
    (r"O\(\s*\w+\s*log\s*\(?\s*\w+\s*\)?\s*\)", 3),
    (r"O\(\s*\w+\^2\s*log\s*\(?\s*\w+\s*\)?\s*\)", 4.5),
    (r"O\(\s*\w+\s*\*\s*\w+\s*\)", 3.5),
    (r"O\(\s*\w+\^2\s*\)", 4),
    (r"O\(\s*\w+\^3\s*\)", 5),
    (r"O\(\s*2\^\w+\s*\)", 6),
    (r"O\(\s*\w+!\s*\)", 7),
]


def question_solution_status(name: str) -> dict | None:
    """Returns {'has_solution': bool, 'accessible': bool} for a problem, using
    authenticated access if credentials are set (so Premium-only editorials
    show as accessible too), or None if the problem isn't found."""
    resp = requests.post(
        "https://leetcode.com/graphql",
        json={"query": _STATUS_QUERY, "variables": {"titleSlug": name}},
        cookies=_COOKIES, headers=_HEADERS, timeout=15,
    )
    data = resp.json().get("data", {}).get("question")
    if data is None:
        return None
    sol = data.get("solution")
    return {"has_solution": bool(sol), "accessible": bool(sol and sol["canSeeDetail"])}


def _fetch_content(name: str) -> str | None:
    resp = requests.post(
        "https://leetcode.com/graphql",
        json={"query": _CONTENT_QUERY, "variables": {"titleSlug": name}},
        cookies=_COOKIES, headers=_HEADERS, timeout=15,
    )
    article = resp.json().get("data", {}).get("ugcArticleOfficialSolutionArticle")
    return article["content"] if article else None


def _split_into_approaches(content: str) -> list[dict]:
    """Splits editorial content on '#### Approach[ N]:' headers, returning one
    dict per approach with its 1-based index, the uuid of its first embedded
    playground (if any), and its raw text chunk (for complexity parsing).
    Handles both the numbered convention ("Approach 1: ...", used when an
    editorial presents several approaches) and the unnumbered single-approach
    convention ("Approach: ...", used when there's just one) -- e.g.
    sliding-window-maximum's editorial has exactly one "### Approach:
    Monotonic Deque" header with no number at all."""
    header_re = re.compile(r"^#+\s*Approach(?:\s+(\d+))?\s*:", re.MULTILINE)
    headers = list(header_re.finditer(content))
    approaches = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
        chunk = content[start:end]
        uuids = re.findall(r"playground/([A-Za-z0-9]+)/shared", chunk)
        index = int(m.group(1)) if m.group(1) else i + 1
        approaches.append({"index": index, "uuid": uuids[0] if uuids else None, "text": chunk})
    return approaches


def _parse_recommended_indices(content: str) -> list[int]:
    """Looks for a "we recommend Approach N ... [and Approach M ...]"
    sentence (typically in an Overview section) and returns the recommended
    approach indices in the order mentioned, or [] if no such sentence is
    found. The span between "Approach N" mentions is often filled with a
    descriptive title (e.g. "recommend *Approach 1: Dynamic Programming +
    Deque* and *Approach 4: ... (Compressed)* since they have the best
    performance"), so this grabs every "Approach \\d+" mention within a
    bounded window after "recommend" rather than expecting a clean digit list
    immediately following it."""
    m = re.search(r"recommend\b", content, re.IGNORECASE)
    if not m:
        return []
    tail = content[m.end():m.end() + 400]
    end_match = re.search(r"\.\s", tail)
    window = tail[: end_match.start() + 1] if end_match else tail
    return [int(x) for x in re.findall(r"Approach\s+(\d+)", window)]


def _complexity_rank(text: str) -> float | None:
    """Finds a 'Time complexity: $$...$$' line and ranks it via
    _COMPLEXITY_PATTERNS, or None if no line is found or it doesn't match a
    recognized pattern."""
    m = re.search(r"Time complexity:.*?\$\$(.*?)\$\$", text, re.DOTALL)
    if not m:
        return None
    expr = m.group(1)
    for pattern, rank in _COMPLEXITY_PATTERNS:
        if re.search(pattern, expr):
            return rank
    return None


def select_best_approach(content: str) -> dict | None:
    """Applies the recommend-text -> complexity-comparison -> last-approach
    priority chain to pick one approach dict (see _split_into_approaches),
    or None if the editorial has no parseable approaches at all."""
    approaches = _split_into_approaches(content)
    if not approaches:
        return None

    recommended = _parse_recommended_indices(content)
    if recommended:
        by_index = {a["index"]: a for a in approaches}
        chosen = by_index.get(recommended[0])
        if chosen and chosen["uuid"]:
            return chosen

    ranked = [(a, _complexity_rank(a["text"])) for a in approaches]
    ranked = [(a, r) for a, r in ranked if r is not None and a["uuid"]]
    if ranked:
        return min(ranked, key=lambda pair: pair[1])[0]

    for a in reversed(approaches):
        if a["uuid"]:
            return a
    return None


def fetch_official_python_solution(name: str) -> str | None:
    """Fetches the editorial for `name`, picks the best approach via
    select_best_approach, and returns its python3 code -- or None if there's
    no editorial, no parseable approach, or no python3 variant for the chosen
    approach's playground."""
    content = _fetch_content(name)
    if not content:
        return None
    approach = select_best_approach(content)
    if not approach:
        return None

    resp = requests.post(
        "https://leetcode.com/graphql",
        json={"query": _PLAYGROUND_QUERY, "variables": {"uuid": approach["uuid"]}},
        cookies=_COOKIES, headers=_HEADERS, timeout=15,
    )
    codes = resp.json().get("data", {}).get("allPlaygroundCodes") or []
    for c in codes:
        if c["langSlug"] == "python3":
            return c["code"]
    return None
