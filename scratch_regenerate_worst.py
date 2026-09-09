import pickle
import time

import categorical_similarity as cs
import evaluate_categorical_similarity as ev
import test_approach_abstract as taa

NAMES = [
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
assert len(NAMES) == 64

df = cs.load_dataset()
with open("abstract_discovery_cache.pkl", "rb") as f:
    abstracts = pickle.load(f)

# keep the old versions for a before/after diff
old_versions = {n: abstracts.get(n) for n in NAMES}
with open("/tmp/old_worst_offender_abstracts.pkl", "wb") as f:
    pickle.dump(old_versions, f)

sample = df[df["name"].isin(NAMES)]
failures = {}
n_done = 0
for _, problem in sample.iterrows():
    name = problem["name"]
    try:
        abstracts[name] = taa.get_abstract(problem[["name", "description", "code"]].to_dict())
        n_done += 1
        if n_done % 15 == 0:
            with open("abstract_discovery_cache.pkl", "wb") as f:
                pickle.dump(abstracts, f)
            print(f"  ...{n_done}/{len(NAMES)} regenerated")
    except Exception as e:
        failures[name] = str(e)
        print(f"  [failed] {name}: {str(e)[:200]}")

with open("abstract_discovery_cache.pkl", "wb") as f:
    pickle.dump(abstracts, f)

print(f"\nDone. {n_done}/{len(NAMES)} regenerated, {len(failures)} failures.")
if failures:
    print("Failed:", list(failures.keys()))
