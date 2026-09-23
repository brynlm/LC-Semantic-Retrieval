"""
Shared building blocks for the new-problem pipeline (discover_new_problems.py
-> fetch_new_problem_metadata.py -> fetch_new_problem_editorials.py ->
fetch_new_problem_community_solutions.py -> build_new_problem_source_cache.py
-> generate_new_problem_abstracts.py -> merge_new_problems_into_production.py
-> export_static_data.py). Each function here was originally written inside
one of that pipeline's own exploratory scripts and is promoted here because
later stages import it directly -- it's load-bearing, not scratch work.

  - anonymize_code(): AST-based identifier anonymization, applied to every
    Python solution before it reaches the LLM (see generate_new_problem_
    abstracts.py). Originally in experiments/scratch_code_embedding_test.py.
  - is_usable_solution_code(): guards against incomplete/prose-only cached
    "solutions". Originally in experiments/structural_features.py -- only
    this one function from that module is production-relevant; the rest
    (dense structural feature vectors, motif detection) was a separate,
    still-experimental research direction and stays in experiments/.
  - leaks() / scrub_leaks(): safety net that catches an LLM-generated
    abstract leaking a problem's own domain vocabulary (e.g. "stone",
    "prerequisite") even when the input code was anonymized. Originally in
    experiments/scratch_code_only_abstracts.py.
  - SYSTEM_PROMPT / _parse_analysis(): the actual prompt contract and
    response parser used by every abstract-generation run. Originally in
    test_approach_abstract.py, which also carries a multi-provider
    (ollama/groq/anthropic) manual-testing harness and an unconditional,
    network-fetching module-level dataset load -- neither belongs in a
    module other scripts import for production use, so only the prompt and
    parser were promoted here; the harness stays in
    experiments/test_approach_abstract.py.
"""

import ast
import json
import re

# --- code anonymization -----------------------------------------------------


class _Anonymizer(ast.NodeTransformer):
    """Renames the function name and every local variable/parameter to a
    generic positional placeholder based on first appearance. Leaves
    attribute names (`.append`, `.heappush`), imported module names
    (`heapq`, `collections`), builtin calls (`len`, `range`, `sorted`), and
    all literals untouched -- only names that are ASSIGNED to or used as
    parameters within this function are considered "local" and renamed."""

    def __init__(self):
        self.rename_map = {}
        self.counter = 0

    def _get_placeholder(self, name):
        if name in ("self", "cls"):
            return name
        if name not in self.rename_map:
            self.rename_map[name] = f"v{self.counter}"
            self.counter += 1
        return self.rename_map[name]

    def visit_FunctionDef(self, node):
        node.name = "solve"
        for arg in node.args.args:
            if arg.arg not in ("self", "cls"):
                arg.arg = self._get_placeholder(arg.arg)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node


def anonymize_code(code: str) -> str:
    """First pass: collect every assigned/for-target/with-target name across
    the whole tree (so a name used before its own assignment in a later
    branch still gets mapped consistently), then rename. Falls back to the
    original code on a parse error."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    anonymizer = _Anonymizer()
    # pre-seed the rename map by walking assignment targets in order, so
    # names get consistent placeholders even before visit_Name reaches them
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.comprehension)):
            targets = [node.target] if isinstance(node, ast.For) else [node.target]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    anonymizer._get_placeholder(n.id)

    new_tree = anonymizer.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        return code


# --- solution-code validity check -------------------------------------------


def is_usable_solution_code(code: str) -> bool:
    """True if `code` parses and contains at least one function/method def.
    Guards against a real data-quality issue found in
    official_editorial_code_cache.pkl: a handful of entries are incomplete
    prose fragments scraped from editorial text -- a bare sequence of
    top-level statements with no function wrapper at all, not a real
    solution. Callers should check this before trusting cached editorial
    code, rather than trusting cache presence alone."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))


# --- narrative-leakage safety net -------------------------------------------

BANNED_WORDS = ["stone", "pile", "course", "prerequisite", "semester", "student", "coin",
                "ghost", "building", "cat", "mouse", "senate", "atom", "potion"]

# Generic, technique-neutral drop-in replacements for each banned word. These
# exist so a leak can be patched with a single regex substitution instead of
# a full retry -- confirmed (see parallel-courses-ii) that the model isn't
# reading these off any identifier in the code, it's just the vocabulary it
# reaches for to describe the underlying mechanism (e.g. "prerequisite" for
# a bitmask dependency check even when no code identifier says that word at
# all), so retrying with a different model doesn't reliably avoid it either
# -- swapping the word after the fact is both cheaper and more reliable.
SAFE_SYNONYMS = {
    "stone": "element", "pile": "group", "course": "task",
    "prerequisite": "dependency", "semester": "round", "student": "participant",
    "coin": "token", "ghost": "pursuer", "building": "node",
    "cat": "pursuer", "mouse": "evader", "senate": "faction",
    "atom": "unit", "potion": "item",
}


def leaks(mechanism: str) -> list:
    """Whole-word match, plural-aware -- a naive substring check
    false-positives on "cat" in "concatenate", "pile" in "compile"/
    "compilation", "atom" in "atomic", "stone" in "milestone"/"keystone",
    "course" in "recourse"/"discourse", etc., which wastes retries (and
    rate-limit budget) rejecting abstracts that never actually leaked
    anything -- but a bare \\bword\\b also misses the plural ("stones")."""
    text = mechanism.lower()
    return [w for w in BANNED_WORDS if re.search(rf"\b{re.escape(w)}e?s?\b", text)]


def scrub_leaks(mechanism: str, leaked_words: list) -> str:
    """Whole-word, case- and plural-preserving substitution of each leaked
    word for its SAFE_SYNONYMS entry. Cheaper and more reliable than a
    retry -- see the SAFE_SYNONYMS comment for why retrying doesn't
    reliably dodge these."""
    text = mechanism
    for w in leaked_words:
        repl = SAFE_SYNONYMS[w]

        def _sub(m, repl=repl):
            matched = m.group(0)
            r = repl + "s" if matched.lower().endswith("s") else repl
            return r.capitalize() if matched[0].isupper() else r

        text = re.sub(rf"\b{re.escape(w)}e?s?\b", _sub, text, flags=re.IGNORECASE)
    return text


# --- abstract-generation prompt contract ------------------------------------

SYSTEM_PROMPT = """You analyze competitive-programming solutions to extract their \
algorithmic structure: the individual techniques involved, and the mechanism \
that ties them together.

Given a problem's description and a working solution, respond with ONLY a \
single JSON object (no markdown code fences, no commentary before or after) \
in exactly this shape:

{
  "techniques": [
    {
      "name": "<short technique/data-structure name, e.g. \\"heap\\", \\"hash set\\", \\"prefix sum\\">",
      "role": "<1 sentence, in plain language, on what this technique accomplishes in this solution. For niche or compound techniques, explain the underlying mechanism itself rather than just naming it -- e.g. for lazy deletion from a heap: \\"stale entries are left in place and skipped over lazily when they surface at the top, rather than being removed immediately on invalidation.\\">"
    }
  ],
  "mechanism": "<2-4 sentences describing how the techniques above work together to solve the problem. Write this around the underlying goal/invariant (e.g. maintaining a running target sum, finding the nearest valid boundary) rather than leading with any single technique's name. This field must stand alone and read clearly without the techniques list for context -- it is what gets embedded for similarity search.>"
}

Rules:
- List at most 3 techniques -- the ones that are genuinely distinct and \
  load-bearing for the solution. Do not rank or single out a "primary" one, \
  but do NOT list implementation substeps as if they were separate \
  techniques (e.g. "index increment" or "variable initialization" are not \
  techniques). If several steps are really just parts of one cohesive \
  technique (e.g. scanning a string character-by-character while \
  maintaining running state), describe that as ONE technique, not several.
- Translate every problem-specific noun into the bare computer-science object \
  it actually is, and write the whole "mechanism" field using ONLY that \
  vocabulary -- "graph", "node", "directed edge", "interval", "array", \
  "sequence", "tree", "set", "state", "connected component", and so on. Never \
  use the problem's own nouns (course, semester, student, letter, coin, ghost, \
  building, cat, mouse, senate, atom, potion, stone, pile, prerequisite) or \
  thin rewordings of them -- e.g. "tasks with prerequisite dependencies" is \
  still the problem's own narrative dressed up; the actual CS object is \
  "nodes with precedence edges." This applies even when a problem is one of \
  a same-title series (e.g. "Stone Game", "Parallel Courses", "Course \
  Schedule" I/II/III...) -- reusing that series' flavor noun (stone, pile, \
  course) makes every entry in the series look artificially similar to every \
  other regardless of whether their actual techniques match, which is \
  exactly the failure mode this rule exists to prevent. Two problems that \
  apply the same technique to the same kind of object should read in \
  near-identical CS vocabulary even when their surface stories have nothing \
  in common. For example:
    BAD:  "tasks that must be completed in an order respecting prerequisite \
          dependencies between them"
    GOOD: "nodes in a directed graph that must be ordered so every edge \
          points from an earlier node to a later one"
    BAD:  "the number of ways to climb a staircase by advancing one or two \
          steps at a time"
    GOOD: "the number of ways to reach a target position in a sequence \
          where each move advances by one of a small fixed set of offsets"
    BAD:  "determining the alphabetical order implied by an alien \
          language's word list"
    GOOD: "recovering a total order over a set of symbols from pairwise \
          precedence constraints"
- Do not name the specific problem or reuse phrasing copied from the problem \
  statement -- this includes synonyms and paraphrases of its entities, not \
  just its exact words.
- "mechanism" must not say things like "as described above" or otherwise \
  depend on the techniques list to make sense.
- Output ONLY the JSON object. No markdown fences, no leading or trailing text.
"""


def _parse_analysis(raw_text: str) -> dict:
    """
    Models don't always obey "no markdown fences" perfectly, so this pulls
    out the first {...} block rather than assuming raw_text is pure JSON.
    They also occasionally emit almost-valid JSON -- e.g. a stray backslash
    before the closing quote of the last string field, which makes the
    parser read it as an escaped quote and keep scanning for a real
    terminator that never comes ("Unterminated string"). Strict json.loads
    is tried first (cheap, handles the common case); only on failure do we
    pay for json_repair, which is built specifically for this class of
    almost-valid LLM JSON output. If neither works, the output is validated
    the same way either way, so a genuinely malformed/incomplete response
    still surfaces a clear error instead of silently returning garbage.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {raw_text!r}")
    candidate = text[start : end + 1]

    try:
        analysis = json.loads(candidate)
    except json.JSONDecodeError as strict_err:
        try:
            import json_repair

            analysis = json_repair.loads(candidate)
        except Exception:
            raise ValueError(
                f"Model output wasn't valid JSON ({strict_err}), and "
                f"json_repair couldn't recover it either: {raw_text!r}"
            ) from strict_err
        if not isinstance(analysis, dict):
            raise ValueError(
                f"json_repair recovered non-dict output ({type(analysis)}): "
                f"{raw_text!r}"
            )
        print("  [warning: JSON needed repair -- worth tracking how often "
              "this happens across the full pool]")

    techniques = analysis.get("techniques")
    if not isinstance(techniques, list) or not techniques:
        raise ValueError(f"Missing/empty 'techniques' list: {analysis!r}")
    for t in techniques:
        if not isinstance(t, dict) or not t.get("name") or not t.get("role"):
            raise ValueError(f"Malformed technique entry: {t!r}")

    mechanism = analysis.get("mechanism")
    if not isinstance(mechanism, str) or not mechanism.strip():
        raise ValueError(f"Missing/empty 'mechanism': {analysis!r}")

    return analysis
