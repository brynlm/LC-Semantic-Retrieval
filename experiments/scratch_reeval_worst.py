import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import evaluate_categorical_similarity as ev

REGENERATED = [
    "add-binary", "add-two-numbers", "alien-dictionary",
    "array-with-elements-not-equal-to-average-of-neighbors",
    "best-time-to-buy-and-sell-stock", "binary-tree-inorder-traversal",
    "capacity-to-ship-packages-within-d-days", "climbing-stairs",
    "closest-binary-search-tree-value-ii", "count-good-triplets-in-an-array",
    "count-of-range-sum", "count-of-smaller-numbers-after-self",
    "count-subarrays-with-fixed-bounds", "count-univalue-subtrees",
    "course-schedule", "course-schedule-ii",
    "create-sorted-array-through-instructions", "decode-string",
    "describe-the-painting", "different-ways-to-add-parentheses",
    "distinct-echo-substrings", "fibonacci-number",
    "find-substring-with-given-hash-value",
    "find-the-longest-valid-obstacle-course-at-each-position",
    "gcd-sort-of-an-array", "how-many-numbers-are-smaller-than-the-current-number",
    "jump-game-vi", "k-closest-points-to-origin",
    "longest-continuous-increasing-subsequence",
    "longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit",
    "longest-happy-prefix", "longest-increasing-subsequence",
    "longest-univalue-path", "maximum-deletions-on-a-string",
    "maximum-subarray", "min-cost-climbing-stairs",
    "minimum-cost-of-a-path-with-special-roads",
    "minimum-distance-between-bst-nodes", "minimum-height-trees",
    "minimum-time-to-revert-word-to-initial-state-i",
    "minimum-time-to-revert-word-to-initial-state-ii", "n-th-tribonacci-number",
    "number-of-atoms", "number-of-connected-components-in-an-undirected-graph",
    "number-of-longest-increasing-subsequence", "number-of-provinces",
    "number-of-restricted-paths-from-first-to-last-node",
    "number-of-ways-to-arrive-at-destination", "path-with-maximum-probability",
    "rank-transform-of-a-matrix", "reverse-nodes-in-k-group", "reverse-pairs",
    "sequence-reconstruction", "shifting-letters-ii", "shortest-palindrome",
    "sliding-window-maximum", "sort-characters-by-frequency",
    "split-array-largest-sum", "sum-of-scores-of-built-strings",
    "swap-nodes-in-pairs", "the-score-of-students-solving-math-expression",
    "top-k-frequent-elements", "top-k-frequent-words", "wiggle-sort",
]

df = cs.load_dataset()
with open("abstract_discovery_cache.pkl", "rb") as f:
    abstracts = pickle.load(f)
with open("mechanism_embedding_cache.pkl", "rb") as f:
    mech_emb = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
mechanisms = [abstracts[n]["mechanism"] for n in REGENERATED]
vectors = model.encode(mechanisms, normalize_embeddings=True)
for n, v in zip(REGENERATED, vectors):
    mech_emb[n] = v
with open("mechanism_embedding_cache.pkl", "wb") as f:
    pickle.dump(mech_emb, f)

pool = sorted(set(abstracts) & set(mech_emb) & set(df["name"]))
names = pool
all_vectors = np.stack([mech_emb[n] for n in names])
all_vectors = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
cos_sim = all_vectors @ all_vectors.T
mechanism_sim = pd.DataFrame(cos_sim, index=names, columns=names)


def rank_of(a, b):
    return mechanism_sim[a].drop(index=a).rank(ascending=False)[b]


# the 37 previously-poor LEGITIMATE pairs with their OLD worst_rank
poor_pairs_old = [
    ("minimum-height-trees", "course-schedule-ii", 441),
    ("minimum-height-trees", "course-schedule", 419),
    ("course-schedule-ii", "alien-dictionary", 336),
    ("longest-increasing-subsequence", "find-the-longest-valid-obstacle-course-at-each-position", 321),
    ("array-with-elements-not-equal-to-average-of-neighbors", "wiggle-sort", 298),
    ("binary-tree-inorder-traversal", "closest-binary-search-tree-value-ii", 232),
    ("sequence-reconstruction", "course-schedule-ii", 189),
    ("count-subarrays-with-fixed-bounds", "longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit", 162),
    ("climbing-stairs", "n-th-tribonacci-number", 155),
    ("k-closest-points-to-origin", "top-k-frequent-words", 143),
    ("capacity-to-ship-packages-within-d-days", "split-array-largest-sum", 126),
    ("climbing-stairs", "fibonacci-number", 121),
    ("describe-the-painting", "shifting-letters-ii", 118),
    ("find-substring-with-given-hash-value", "distinct-echo-substrings", 115),
    ("minimum-time-to-revert-word-to-initial-state-i", "longest-happy-prefix", 114),
    ("rank-transform-of-a-matrix", "gcd-sort-of-an-array", 112),
    ("maximum-subarray", "best-time-to-buy-and-sell-stock", 112),
    ("longest-continuous-increasing-subsequence", "number-of-longest-increasing-subsequence", 101),
    ("path-with-maximum-probability", "number-of-ways-to-arrive-at-destination", 101),
    ("longest-univalue-path", "count-univalue-subtrees", 83),
    ("reverse-pairs", "count-of-smaller-numbers-after-self", 75),
    ("decode-string", "number-of-atoms", 74),
    ("different-ways-to-add-parentheses", "the-score-of-students-solving-math-expression", 57),
    ("top-k-frequent-elements", "sort-characters-by-frequency", 46),
    ("count-good-triplets-in-an-array", "create-sorted-array-through-instructions", 40),
    ("minimum-distance-between-bst-nodes", "binary-tree-inorder-traversal", 38),
    ("reverse-pairs", "count-of-range-sum", 37),
    ("minimum-cost-of-a-path-with-special-roads", "number-of-restricted-paths-from-first-to-last-node", 35),
    ("shortest-palindrome", "maximum-deletions-on-a-string", 34),
    ("sum-of-scores-of-built-strings", "longest-happy-prefix", 31),
    ("jump-game-vi", "sliding-window-maximum", 27),
    ("number-of-provinces", "number-of-connected-components-in-an-undirected-graph", 27),
    ("climbing-stairs", "min-cost-climbing-stairs", 26),
    ("longest-happy-prefix", "minimum-time-to-revert-word-to-initial-state-ii", 25),
    ("count-of-smaller-numbers-after-self", "how-many-numbers-are-smaller-than-the-current-number", 24),
    ("swap-nodes-in-pairs", "reverse-nodes-in-k-group", 23),
    ("add-binary", "add-two-numbers", 21),
]

print(f"{'pair':90s} {'old':>6s} {'new_ab':>7s} {'new_ba':>7s} {'new_worst':>10s} {'delta':>7s}")
improved, worse, same_ish = 0, 0, 0
for a, b, old_rank in poor_pairs_old:
    rank_ab = rank_of(a, b)
    rank_ba = rank_of(b, a)
    new_worst = max(rank_ab, rank_ba)
    delta = old_rank - new_worst
    tag = "IMPROVED" if delta > 5 else ("WORSE" if delta < -5 else "~same")
    if tag == "IMPROVED":
        improved += 1
    elif tag == "WORSE":
        worse += 1
    else:
        same_ish += 1
    print(f"{a[:44]+'/'+b[:44]:90s} {old_rank:6d} {rank_ab:7.0f} {rank_ba:7.0f} {new_worst:10.0f} {delta:+7.0f}  {tag}")

print(f"\nImproved (>5 rank gain): {improved}, Worse (>5 rank loss): {worse}, ~same: {same_ish}")
print(f"Median old worst_rank: {np.median([p[2] for p in poor_pairs_old]):.0f}")
print(f"Median new worst_rank: {np.median([max(rank_of(a,b), rank_of(b,a)) for a,b,_ in poor_pairs_old]):.0f}")
