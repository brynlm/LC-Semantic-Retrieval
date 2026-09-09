"""
Tests whether embedding solution code directly (raw, or with identifiers
anonymized) does better or worse than the LLM-generated mechanism-abstract
embeddings at (a) avoiding narrative/name leakage and (b) actually tracking
real technique similarity -- validated against the 639 hand-judged verdicts
from the manual quality audit, not just eyeballed on one cluster.

Three code representations tested, all embedded with the same
all-MiniLM-L6-v2 model already used for mechanism abstracts (a general NL
model, not code-specialized -- a real caveat noted in the report, not hidden):
  1. raw code, as-is (includes the LeetCode-canonical function name -- e.g.
     `stoneGameII` -- and original variable names like `piles`/`stones`).
  2. identifier-anonymized code: function name and all local
     variables/parameters renamed to generic positional placeholders
     (v0, v1, ... in first-appearance order), keeping control flow, stdlib
     calls (heapq, deque, ...), attribute/method names, and literals intact.

Usage:
    python scratch_code_embedding_test.py
"""

import ast
import csv
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import categorical_similarity as cs

_BUILTIN_ATTRS_KEEP = None  # placeholder, not used -- attribute names are always kept


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


def main():
    df = cs.load_dataset()
    name_to_code = dict(zip(df["name"], df["code"]))
    with open("official_editorial_code_cache.pkl", "rb") as f:
        official_code = pickle.load(f)

    with open("/tmp/audit_verdicts.csv") as fcsv:
        audit_rows = list(csv.DictReader(fcsv))
    seen = set()
    dedup = []
    for r in audit_rows:
        if r["query"] in seen:
            continue
        seen.add(r["query"])
        dedup.append(r)

    names_needed = set()
    for r in dedup:
        names_needed.add(r["query"])
        names_needed.add(r["neighbor"])
    names_needed &= set(name_to_code)
    print(f"Names needed for code embedding: {len(names_needed)}")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    raw_code = {}
    anon_code = {}
    for n in names_needed:
        code = official_code.get(n, name_to_code[n])
        raw_code[n] = code
        anon_code[n] = anonymize_code(code)

    names_list = sorted(names_needed)
    raw_vecs = embed_model.encode([raw_code[n] for n in names_list], normalize_embeddings=True, show_progress_bar=True)
    anon_vecs = embed_model.encode([anon_code[n] for n in names_list], normalize_embeddings=True, show_progress_bar=True)

    raw_sim = pd.DataFrame(raw_vecs @ raw_vecs.T, index=names_list, columns=names_list)
    anon_sim = pd.DataFrame(anon_vecs @ anon_vecs.T, index=names_list, columns=names_list)
    raw_sim.to_pickle("/tmp/raw_code_sim.pkl")
    anon_sim.to_pickle("/tmp/anon_code_sim.pkl")

    print("\n=== Stone Game family, RAW code similarity ===")
    sg = sorted(n for n in names_list if n.startswith("stone-game"))
    print(raw_sim.loc[sg, sg].round(3).to_string())
    print("\n=== Stone Game family, ANONYMIZED code similarity ===")
    print(anon_sim.loc[sg, sg].round(3).to_string())

    print("\n=== Validation against the 639 hand-judged verdicts ===")
    for label, sim in [("raw code", raw_sim), ("anonymized code", anon_sim)]:
        by_verdict = {"match": [], "partial": [], "mismatch": []}
        for r in dedup:
            q, n, v = r["query"], r["neighbor"], r["verdict"]
            if q in sim.index and n in sim.columns:
                by_verdict[v].append(sim.loc[q, n])
        print(f"\n{label}:")
        for v in ("match", "partial", "mismatch"):
            arr = np.array(by_verdict[v])
            if len(arr):
                print(f"  {v:9s} n={len(arr):4d}  mean={arr.mean():.3f}  median={np.median(arr):.3f}")

    print("\n(for reference) mechanism-abstract similarity, same breakdown:")
    with open("mechanism_embedding_cache.pkl", "rb") as f:
        mech_emb = pickle.load(f)
    mech_names = [n for n in names_list if n in mech_emb]
    mech_vectors = np.stack([mech_emb[n] for n in mech_names])
    mech_vectors = mech_vectors / np.linalg.norm(mech_vectors, axis=1, keepdims=True)
    mech_sim = pd.DataFrame(mech_vectors @ mech_vectors.T, index=mech_names, columns=mech_names)
    by_verdict = {"match": [], "partial": [], "mismatch": []}
    for r in dedup:
        q, n, v = r["query"], r["neighbor"], r["verdict"]
        if q in mech_sim.index and n in mech_sim.columns:
            by_verdict[v].append(mech_sim.loc[q, n])
    for v in ("match", "partial", "mismatch"):
        arr = np.array(by_verdict[v])
        if len(arr):
            print(f"  {v:9s} n={len(arr):4d}  mean={arr.mean():.3f}  median={np.median(arr):.3f}")


if __name__ == "__main__":
    main()
