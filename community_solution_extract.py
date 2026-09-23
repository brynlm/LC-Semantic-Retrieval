import ast, signal, textwrap
import mistune
import requests

_MD = mistune.create_markdown(renderer=None)

GRAPHQL_URL = "https://leetcode.com/graphql"

_LIST_QUERY = """
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

_CONTENT_QUERY = """
query solutionArticle($topicId: ID!) {
  ugcArticleSolutionArticle(topicId: $topicId) { content }
}
"""


def fetch_candidates(name: str, top_n: int = 8) -> list:
    """Lists a problem's community "Solutions" tab posts, most-voted first.
    Anonymous/public access -- works for any free problem without a session
    cookie; a premium-only problem's candidates require an authenticated
    request instead (see experiments/run_premium_locked_community_fetch.py,
    a one-time pass against the original corpus, not part of the ongoing
    new-problem pipeline)."""
    r = requests.post(
        GRAPHQL_URL,
        json={"query": _LIST_QUERY, "variables": {"questionSlug": name, "skip": 0, "first": top_n}},
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


def fetch_content(topic_id) -> str:
    r = requests.post(GRAPHQL_URL, json={"query": _CONTENT_QUERY, "variables": {"topicId": topic_id}}, timeout=15)
    node = r.json().get("data", {}).get("ugcArticleSolutionArticle")
    content = node.get("content") if node else None
    return content or None


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError()


def normalize_escapes(content):
    """Some posts' content comes back with literal escape sequences
    (backslash-n, backslash-quote) instead of real characters -- looks like
    inconsistent escaping upstream in LeetCode's own stored content, not
    something we control. Detect and fix before markdown parsing."""
    real_newlines = content.count("\n")
    literal_seqs = content.count("\\n")
    if literal_seqs > real_newlines * 2:
        content = (content
                   .replace("\\r\\n", "\n")
                   .replace("\\n", "\n")
                   .replace("\\t", "\t")
                   .replace("\\'", "'")
                   .replace('\\"', '"')
                   .replace("\\\\", "\\"))
    return content


_PY_TAGS = ("python", "py")


def extract_code_blocks_ranked(markdown):
    """Parses markdown via mistune's AST mode (not regex) and returns every
    fenced code block's raw text, ordered so explicitly Python-tagged blocks
    (checked via the parser's own `info` string, not a hand-rolled fence
    regex) come first, then everything else as a fallback -- untagged blocks
    still get tried since some posts label the language only in surrounding
    prose (e.g. a bold "**Python:**" heading) rather than in the fence
    itself."""
    markdown = normalize_escapes(markdown)
    tokens = _MD(markdown)
    py_tagged, other = [], []
    for t in tokens:
        if t.get("type") != "block_code":
            continue
        info = ((t.get("attrs") or {}).get("info") or "").strip().lower()
        raw = t["raw"]
        if any(info.startswith(tag) for tag in _PY_TAGS):
            py_tagged.append(raw)
        else:
            other.append(raw)
    return py_tagged + other


_COMMON_MODULES = (
    "import itertools, collections, functools, math, string, operator, "
    "bisect, heapq, re, sys, random, datetime, string\n"
)


def _try_run(code, prompt, test_src, entry_point, timeout_s=5):
    """Community solutions are untrusted, unverified code -- a buggy or
    pathologically inefficient one (accidental infinite loop, exponential
    blowup on a large test case) must not be able to hang the whole batch.
    signal.alarm bounds wall-clock time per attempt (Unix-only, fine here)."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        namespace = {"xrange": range}  # some highly-voted posts predate Python 3
        # `prompt` does `from itertools import *` etc (unqualified names
        # only) -- community code that writes `itertools.groupby(...)`
        # (qualified) needs the module names bound too.
        exec(_COMMON_MODULES, namespace)
        exec(prompt, namespace)
        exec(code, namespace)
        exec(test_src, namespace)
        candidate = eval(entry_point, namespace)
        namespace["check"](candidate)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _method_name_from_entry_point(entry_point):
    return entry_point.rsplit(".", 1)[-1]


class _RenameNames(ast.NodeTransformer):
    def __init__(self, rename_map):
        self.rename_map = rename_map

    def visit_Name(self, node):
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node


class _RenameParamsInFunction(ast.NodeTransformer):
    def __init__(self, func_name, rename_map):
        self.func_name = func_name
        self.rename_map = rename_map

    def visit_FunctionDef(self, node):
        if node.name != self.func_name:
            self.generic_visit(node)
            return node
        for arg in node.args.args:
            if arg.arg in self.rename_map:
                arg.arg = self.rename_map[arg.arg]
        renamer = _RenameNames(self.rename_map)
        for stmt in node.body:
            renamer.visit(stmt)
        return node


def _align_signature(code, starter_code, method_name):
    try:
        starter_tree = ast.parse(starter_code + "pass")
        code_tree = ast.parse(code)
    except SyntaxError:
        return code

    def find_params(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return [a.arg for a in node.args.args if a.arg != "self"]
        return None

    canonical_params = find_params(starter_tree)
    candidate_params = find_params(code_tree)
    if not canonical_params or not candidate_params:
        return code
    if len(canonical_params) != len(candidate_params):
        return code
    if canonical_params == candidate_params:
        return code

    rename_map = dict(zip(candidate_params, canonical_params))
    _RenameParamsInFunction(method_name, rename_map).visit(code_tree)
    ast.fix_missing_locations(code_tree)
    return ast.unparse(code_tree)


def find_verified_solution(markdown, prompt, test_src, entry_point, starter_code=None):
    needs_class = "Solution()" in entry_point
    method_name = _method_name_from_entry_point(entry_point)

    for raw_code in extract_code_blocks_ranked(markdown):
        code = textwrap.dedent(raw_code)

        candidates_to_try = [code]
        if needs_class and "class Solution" not in code:
            candidates_to_try = ["class Solution:\n" + textwrap.indent(code, "    ")]

        if starter_code:
            base = candidates_to_try[0]
            aligned = _align_signature(base, starter_code, method_name)
            if aligned != base:
                candidates_to_try.append(aligned)

        for cand in candidates_to_try:
            try:
                ast.parse(cand)
                _try_run(cand, prompt, test_src, entry_point)
                return cand
            except Exception:
                continue
    return None
