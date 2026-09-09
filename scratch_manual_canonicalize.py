"""
One-off script: manually canonicalizes the 142 remaining (all count=1) raw
algorithm labels that the LLM canonicalization pass hadn't reached yet,
avoiding further Groq calls entirely. Reuses the 25 canonical names the LLM
already established where they fit; introduces new canonical buckets only
for genuinely distinct techniques.

Pulls the exact raw label strings live from the cache (rather than
hand-retyping them, which risks unicode mismatches -- several use U+2011
non-breaking hyphens) and zips them against a manually-authored ordered
assignment list, in the same descending-count order they were reviewed in.
"""

import pickle
from collections import Counter

with open("phase_taxonomy_discovery_cache.pkl", "rb") as f:
    cache = pickle.load(f)
with open("phase_taxonomy_canonical_mapping.pkl", "rb") as f:
    mapping_by_phase = pickle.load(f)

algo_labels = Counter(e["label"].strip().lower() for t in cache.values() for e in t.get("algorithm", []))
already_mapped = mapping_by_phase.get("algorithm", {})
unmapped_ordered = [label for label, _ in sorted(algo_labels.items(), key=lambda kv: -kv[1]) if label not in already_mapped]

assert len(unmapped_ordered) == 142, f"expected 142 unmapped, got {len(unmapped_ordered)} -- order/content shifted, re-verify"

# Manually assigned canonical label for each of the 142, in the exact same
# order as unmapped_ordered (descending count, ties in Counter-stable order).
manual_canonicals = [
    "Line Sweep",                              # sweep line with min-heap
    "Hash Table",                              # bucket hashing
    "Heap (Priority Queue)",                   # heap selection
    "Dynamic Programming",                     # recursive dp with combinatorics
    "Simulation",                              # constant time conditional
    "Quickselect",                             # quickselect
    "Heap (Priority Queue)",                   # nth largest selection via heap
    "Z-Function",                              # z-function
    "Dynamic Programming",                     # dynamic programming over prefixes
    "Brute Force",                             # brute force substring enumeration
    "Greedy",                                  # greedy matching
    "Two Pointers",                            # two pointers with conditional swapping
    "Fenwick Tree (Binary Indexed Tree)",      # binary indexed tree
    "Fenwick Tree (Binary Indexed Tree)",      # binary indexed tree (fenwick tree)
    "Fenwick Tree (Binary Indexed Tree)",      # fenwick tree (prefix sum counting)
    "Dynamic Programming",                     # top-down dp with memoization
    "Dynamic Programming",                     # monotonic queue dp
    "Linear Scan",                             # single-pass scanning with last positions
    "Simulation",                              # case analysis
    "Bit Manipulation",                        # xor reduction
    "Linear Scan",                             # linear search
    "Hash Table",                              # hash set
    "Binary Search",                           # binary search on length
    "Hash Table",                              # hash set lookup
    "Brute Force",                             # brute force
    "Greedy",                                  # greedy selection of every second element from the middle third
    "Dynamic Programming",                     # memoized recursion (top-down dp)
    "Simulation",                              # conditional check
    "Dynamic Programming",                     # dynamic programming (memoized recursion)
    "Dijkstra's Algorithm",                    # dijkstra's algorithm
    "Dijkstra's Algorithm",                    # dijkstra over extended state space
    "Dynamic Programming",                     # dfs with memoization
    "Linear Scan",                             # linear scan with prefix check
    "Backtracking",                            # backtracking (depth-first search)
    "Dynamic Programming",                     # recursive memoization
    "Depth-First Search",                      # dfs (inorder traversal)
    "Divide and Conquer",                      # recursive tree construction
    "Divide and Conquer",                      # post-order dfs (divide and conquer)
    "Prefix Sum Technique",                    # difference array with prefix sum
    "Greedy",                                  # greedy matching with deque
    "Prefix Sum Technique",                    # prefix sum with running min/max
    "Simulation",                              # rotation check
    "Hash Table",                              # hash comparison
    "Hash Table",                              # hash set with wildcard patterns
    "Cycle Detection",                         # cycle detection
    "Dynamic Programming",                     # topological dp
    "Depth-First Search",                      # dfs with path tracking
    "Topological Sort (Kahn's Algorithm)",     # kahn's algorithm (topological sort via indegree pruning)
    "Prefix Sum Technique",                    # prefix-suffix counting
    "Fenwick Tree (Binary Indexed Tree)",      # fenwick tree (bit)
    "Fenwick Tree (Binary Indexed Tree)",      # fenwick tree (prefix maximum query)
    "Two Pointers",                            # two pointers with binary search
    "Brute Force",                             # nested loop with running maximum
    "Geometry",                                # cross product
    "Simulation",                              # parity check and division
    "Fenwick Tree (Binary Indexed Tree)",      # binary indexed tree (range minimum query)
    "Order-Statistic Tree",                    # order-statistic tree (sortedlist)
    "Fenwick Tree (Binary Indexed Tree)",      # binary indexed tree (fenwick tree) prefix sum
    "Greedy",                                  # factorial-based greedy selection
    "Recursion",                               # recursion
    "Greedy",                                  # greedy decomposition into 3s
    "Dynamic Programming",                     # dp with row/column max
    "Dynamic Programming",                     # dfs with memoization (state compression dp)
    "Dynamic Programming",                     # tree dp with memoization (dfs)
    "Fenwick Tree (Binary Indexed Tree)",      # fenwick tree (prefix sum)
    "Ordered Set",                             # range query in multiset
    "Segment Tree",                            # segment tree (range max query & range assignment)
    "Divide and Conquer",                      # divide and conquer (recursive tree construction)
    "Merge Sort",                              # merge sort counting
    "Divide and Conquer",                      # divide and conquer recursion
    "Brute Force",                             # pairwise comparison
    "Breadth-First Search",                    # bucket queue processing
    "Breadth-First Search",                    # breadth-first search over state space
    "Dynamic Programming",                     # subset dp
    "Dynamic Programming",                     # dynamic programming with memoization
    "Bit Manipulation",                        # binary-to-decimal conversion via bitwise shift
    "Linear Scan",                             # single-pass linked list traversal
    "Number Theory",                           # euclidean gcd
    "Number Theory",                           # divisor enumeration
    "Sieve of Eratosthenes",                   # sieve of eratosthenes
    "Dynamic Programming",                     # dynamic programming over gcd segments
    "Monotonic Stack",                         # monotonic stack
    "Dynamic Programming",                     # interval dp
    "Union-Find",                              # union-find
    "Depth-First Search",                      # dfs bipartite coloring
    "Brute Force",                             # exhaustive search over increment counts
    "Brute Force",                             # brute force submatrix check
    "Simulation",                              # aggregation of counts and zero calculation
    "Bit Manipulation",                        # bitwise manipulation (gray code formula)
    "Backtracking",                            # backtracking
    "Sliding Window",                          # sliding window with frequency counters
    "Depth-First Search",                      # recursive dfs
    "Backtracking",                            # backtracking (dfs)
    "Simulation",                              # filter and count
    "Linear Scan",                             # single scan
    "Dynamic Programming",                     # dynamic programming over graph
    "Breadth-First Search",                    # bfs-based bipartiteness and diameter check
    "Divide and Conquer",                      # recursive divide-and-conquer with combinatorial counting
    "Depth-First Search",                      # post-order depth-first traversal
    "Depth-First Search",                      # in-order traversal
    "Hash Table",                              # hash map with frequency count
    "Recursion",                               # recursive reduction
    "Simulation",                              # direct matrix update
    "Binary Search",                           # heap + binary search assignment
    "Depth-First Search",                      # dfs with top-k pruning
    "Prefix Sum Technique",                    # difference array (prefix sum sweep)
    "Prefix Sum Technique",                    # prefix sum with hash map
    "Dynamic Programming",                     # bottom-up tree dp
    "Lowest Common Ancestor",                  # lowest common ancestor (recursive)
    "Depth-First Search",                      # depth calculation via dfs
    "Depth-First Search",                      # recursive tree traversal (dfs)
    "Cycle Detection",                         # hash set for cycle detection
    "Simulation",                              # digit square sum
    "Depth-First Search",                      # dfs value assignment
    "Topological Sort (Kahn's Algorithm)",     # kahn's algorithm (bfs topological sort)
    "Union-Find",                              # union-find
    "Backtracking",                            # dfs backtracking with pruning
    "Breadth-First Search",                    # breadth-first search on bitmask states
    "Simulation",                              # odd count check
    "Brute Force",                             # brute-force traversal
    "Backtracking",                            # dfs with backtracking
    "Counting",                                # counting
    "Hash Table",                              # hash table lookup in linear scan
    "Binary Search",                           # binary search with greedy feasibility
    "Depth-First Search",                      # dfs
    "Depth-First Search",                      # dfs with time tracking
    "Greedy",                                  # greedy bit manipulation
    "Greedy",                                  # greedy with min-heap
    "Two Pointers",                            # two pointers / greedy matching
    "Heap (Priority Queue)",                   # k-largest selection
    "Depth-First Search",                      # dfs on intersection graph
    "Bit Manipulation",                        # bit manipulation
    "Linear Scan",                             # matrix traversal
    "Greedy",                                  # greedy assignment
    "Simulation",                              # matrix consistency check
    "Dijkstra's Algorithm",                    # dijkstra (per-row and per-column priority queues)
    "Dynamic Programming",                     # dynamic programming over columns
    "Set Operations",                          # set intersection
    "Set Operations",                          # set difference
    "Dynamic Programming",                     # dynamic programming with hash map
    "Hash Table",                              # hash map grouping
    "String Matching",                         # substring search
]

assert len(manual_canonicals) == 142, f"expected 142 assignments, wrote {len(manual_canonicals)}"

manual_mapping = dict(zip(unmapped_ordered, manual_canonicals))
mapping_by_phase["algorithm"] = {**already_mapped, **manual_mapping}

with open("phase_taxonomy_canonical_mapping.pkl", "wb") as f:
    pickle.dump(mapping_by_phase, f)

print(f"algorithm mapping now covers {len(mapping_by_phase['algorithm'])}/{len(algo_labels)} raw labels")
print(f"preprocessing mapping covers {len(mapping_by_phase.get('preprocessing', {}))} raw labels")
