"""
Similarity over labeled categorical metadata only -- no LLM calls, no free-text
embedding. This is the baseline we compare a future hybrid (categorical +
embedding) retrieval system against.

Two signals, combined into one score:
  1. tags       -- list[str] of topics (e.g. "Array", "Hash Table"). Compared
                   with Jaccard similarity, optionally IDF-weighted so that a
                   shared rare tag (e.g. "Suffix Array") counts for more than
                   a shared common one (e.g. "Array").
  2. starter_code -- the function signature is parsed with `ast` into a bag of
                   "arg:<type>" / "ret:<type>" tokens (generics flattened, so
                   List[List[int]] contributes List, List, int) and compared
                   with multiset (Ruzicka) similarity.

difficulty is intentionally not used as a similarity signal -- kept only as a
column for eyeballing results.

Usage:
    import categorical_similarity as cs

    df = cs.load_dataset()
    tag_sim = cs.tag_similarity_matrix(df, idf_weighted=True)
    sig_sim = cs.signature_similarity_matrix(df)
    combined = cs.combine(tag_sim, sig_sim, tag_weight=0.7, sig_weight=0.3)

    cs.top_k_neighbors(combined, df, "two-sum", k=10)
"""

import ast
import os
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

CACHE_PATH = os.path.join(os.path.dirname(__file__), "lc_train_cache.pkl")


def load_dataset(refresh: bool = False) -> pd.DataFrame:
    """
    Loads the full LeetCodeDataset-train split, cached locally as a pickle
    (the HF path has no parquet engine available in this env, and re-fetching
    over the network on every run is slow and needlessly network-dependent).
    Pass refresh=True to re-pull from HF and overwrite the cache.
    """
    if not refresh and os.path.exists(CACHE_PATH):
        return pd.read_pickle(CACHE_PATH)

    df = pd.read_json(
        "hf://datasets/newfacade/LeetCodeDataset/LeetCodeDataset-train.jsonl",
        lines=True,
    )
    df.rename(
        columns={
            "task_id": "name",
            "problem_description": "description",
            "completion": "code",
        },
        inplace=True,
    )
    df.to_pickle(CACHE_PATH)
    return df


def stratified_sample_by_tag(df: pd.DataFrame, per_tag_quota: int = 3, seed: int = 0) -> pd.DataFrame:
    """
    Samples problems for diverse coverage of the tag space, e.g. for an LLM
    labeling pass where you want to see every kind of technique at least a
    few times rather than mostly "Array"/"String" problems.

    Processes tags from rarest to most common, taking up to `per_tag_quota`
    not-yet-selected problems for each. Rarest-first means a rare tag (e.g.
    "Suffix Array", 7 problems total) is guaranteed its quota before common
    tags (e.g. "Array", 1619 problems) have already exhausted the sample with
    problems that happen to co-occur with them.
    """
    rng = np.random.default_rng(seed)
    tag_counts = Counter(t for tags in df["tags"] for t in tags)
    selected_idx: list = []
    selected_set = set()

    for tag, _ in sorted(tag_counts.items(), key=lambda kv: kv[1]):
        candidates = [i for i, tags in zip(df.index, df["tags"]) if tag in tags and i not in selected_set]
        if not candidates:
            continue
        chosen = rng.choice(candidates, size=min(per_tag_quota, len(candidates)), replace=False)
        selected_idx.extend(chosen)
        selected_set.update(chosen)

    return df.loc[selected_idx]


# ---------------------------------------------------------------------------
# Tag similarity
# ---------------------------------------------------------------------------


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dice(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def tag_similarity_matrix(df: pd.DataFrame, idf_weighted: bool = False, metric: str = "jaccard") -> pd.DataFrame:
    """
    Pairwise tag similarity, indexed/columned by problem name.

    idf_weighted=False: plain Jaccard/Dice over the tag sets.
    idf_weighted=True:  each tag's contribution is weighted by its inverse
        document frequency across the dataset, so two problems sharing a
        rare tag (e.g. "Suffix Array") score higher than two sharing only a
        common one (e.g. "Array"). Implemented as weighted-Jaccard (sum(min-
        weight over shared)/sum(max-weight over union)) or weighted-Dice
        (2*sum(weight over shared)/sum(weight over A)+sum(weight over B)),
        both reducing to their plain form when all weights are equal.

    metric="jaccard" (default) or "dice" -- Dice weighs overlap relative to
        each set's own size (2|A∩B|/(|A|+|B|)) rather than the union
        (|A∩B|/|A∪B|), so it's less punishing when one problem has notably
        more tags than the other.
    """
    if metric not in ("jaccard", "dice"):
        raise ValueError(f"Unknown metric: {metric!r}")

    names = df["name"].tolist()
    tag_sets = [set(t) for t in df["tags"]]
    n = len(names)

    if idf_weighted:
        doc_freq = Counter(tag for tags in tag_sets for tag in set(tags))
        idf = {tag: np.log(n / df_count) + 1.0 for tag, df_count in doc_freq.items()}
    else:
        idf = None

    sim = np.eye(n)
    for i, j in combinations(range(n), 2):
        a, b = tag_sets[i], tag_sets[j]
        if idf is None:
            s = _jaccard(a, b) if metric == "jaccard" else _dice(a, b)
        else:
            shared = a & b
            if metric == "jaccard":
                union = a | b
                s = sum(idf[t] for t in shared) / sum(idf[t] for t in union) if union else 0.0
            else:
                denom = sum(idf[t] for t in a) + sum(idf[t] for t in b)
                s = 2 * sum(idf[t] for t in shared) / denom if denom else 0.0
        sim[i, j] = sim[j, i] = s

    return pd.DataFrame(sim, index=names, columns=names)


def difficulty_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise difficulty similarity: 1.0 for an exact match, 0.5 for adjacent
    (Easy/Medium or Medium/Hard), 0.0 for Easy/Hard. Kept separate from
    tag_similarity_matrix since it was assumed low-value going in -- this
    lets that assumption actually be tested against the ground-truth eval
    rather than taken on faith.
    """
    names = df["name"].tolist()
    order = {"Easy": 0, "Medium": 1, "Hard": 2}
    levels = df["difficulty"].map(order).to_numpy()
    dist = np.abs(levels[:, None] - levels[None, :])
    sim = 1.0 - 0.5 * dist
    return pd.DataFrame(sim, index=names, columns=names)


# ---------------------------------------------------------------------------
# Signature similarity (parsed from starter_code)
# ---------------------------------------------------------------------------

# starter_code bodies are stubs ("...:\n        ") with no statement after the
# colon, which is a SyntaxError on its own -- pad with an indented `pass` at a
# few indentation depths until one parses.
_PASS_SUFFIXES = ["\npass", "\n    pass\n", "\n        pass\n", "\n            pass\n"]


def _parse_module(starter_code: str) -> ast.Module | None:
    for suffix in _PASS_SUFFIXES:
        try:
            return ast.parse(starter_code + suffix)
        except SyntaxError:
            continue
    return None


def _flatten_annotation(node: ast.AST | None) -> list[str]:
    """Flattens a (possibly generic) type annotation into its base tokens,
    e.g. Optional[List[int]] -> ["Optional", "List", "int"]."""
    if node is None:
        return []
    tokens = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            tokens.append(n.id)
        elif isinstance(n, ast.Attribute):
            tokens.append(n.attr)
        elif isinstance(n, ast.Constant) and n.value is None:
            tokens.append("None")
    return tokens


def parse_signature(starter_code: str, entry_point: str) -> dict | None:
    """
    Extracts the entry-point method's parameter/return type tokens from
    starter_code. entry_point looks like "Solution().twoSum"; only the method
    name is used since every problem in this dataset defines a single
    top-level `class Solution`.

    Returns {"arg_tokens": [...], "ret_tokens": [...], "n_args": int} or None
    if the method/annotations can't be recovered (e.g. no return annotation
    at all is common and fine -- only outright parse failure returns None).
    """
    method_name = entry_point.split(".")[-1].split("(")[0]
    tree = _parse_module(starter_code)
    if tree is None:
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            args = [a for a in node.args.args if a.arg != "self"]
            arg_tokens = [tok for a in args for tok in _flatten_annotation(a.annotation)]
            ret_tokens = _flatten_annotation(node.returns)
            return {"arg_tokens": arg_tokens, "ret_tokens": ret_tokens, "n_args": len(args)}

    return None


def _signature_bag(sig: dict) -> Counter:
    bag = Counter(f"arg:{t}" for t in sig["arg_tokens"])
    bag.update(f"ret:{t}" for t in sig["ret_tokens"])
    bag[f"nargs:{sig['n_args']}"] += 1
    return bag


def _ruzicka_similarity(a: Counter, b: Counter) -> float:
    """Multiset (weighted Jaccard) similarity: sum(min)/sum(max) over the
    union of keys. Reduces to plain Jaccard when counts are all 0/1."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    num = sum(min(a[k], b[k]) for k in keys)
    den = sum(max(a[k], b[k]) for k in keys)
    return num / den if den else 0.0


def signature_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise signature similarity, indexed/columned by problem name.
    Rows whose signature failed to parse get an all-zero similarity row/col
    (they're printed as a warning so parse coverage stays visible)."""
    names = df["name"].tolist()
    sigs = [parse_signature(sc, ep) for sc, ep in zip(df["starter_code"], df["entry_point"])]
    n_failed = sum(s is None for s in sigs)
    if n_failed:
        print(f"[signature_similarity_matrix] {n_failed}/{len(sigs)} signatures failed to parse")
    bags = [_signature_bag(s) if s is not None else Counter() for s in sigs]

    n = len(names)
    sim = np.eye(n)
    for i, j in combinations(range(n), 2):
        s = _ruzicka_similarity(bags[i], bags[j])
        sim[i, j] = sim[j, i] = s

    return pd.DataFrame(sim, index=names, columns=names)


# ---------------------------------------------------------------------------
# Combining signals
# ---------------------------------------------------------------------------


def combine(
    tag_sim: pd.DataFrame,
    sig_sim: pd.DataFrame,
    tag_weight: float = 0.8,
    sig_weight: float = 0.2,
    difficulty_sim: pd.DataFrame | None = None,
    difficulty_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Defaults (tag_weight=0.8, sig_weight=0.2) come from a weight sweep against
    the full ground-truth eval (evaluate_categorical_similarity.py), not a
    guess -- 0.8 was a clear, stable peak for both Jaccard and Dice tag
    similarity (hit_rate@10 ~0.364 vs ~0.358 at the original arbitrary 0.7/0.3
    guess). difficulty_weight defaults to 0.0 since its measured benefit was
    small (~+0.002 hit_rate@10 at weight ~0.10-0.15) -- pass difficulty_sim
    (from difficulty_similarity_matrix) and a small difficulty_weight to
    include it; the improvement is real but modest enough that it's opt-in
    rather than default.
    """
    result = tag_weight * tag_sim + sig_weight * sig_sim
    if difficulty_sim is not None and difficulty_weight:
        result = (1 - difficulty_weight) * result + difficulty_weight * difficulty_sim
    return result


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------


def top_k_neighbors(sim: pd.DataFrame, df: pd.DataFrame, name: str, k: int = 10) -> pd.DataFrame:
    """Nearest neighbors of `name` by similarity, with tags/difficulty joined
    in for eyeballing whether they actually relate."""
    meta = df.set_index("name")[["tags", "difficulty"]]
    scores = sim[name].drop(index=name).sort_values(ascending=False).head(k)
    return scores.to_frame("similarity").join(meta)
