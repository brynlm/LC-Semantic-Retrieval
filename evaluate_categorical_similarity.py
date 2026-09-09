"""
Evaluation for categorical_similarity.py.

Primary evaluation is against real ground truth: LeetCode's own "Similar
Questions" edges, scraped in the kaysss/leetcode-problem-detailed HF dataset.
This is LeetCode's own editorial judgment, not something we invented to grade
our own similarity measure -- see `load_similar_questions_ground_truth`.

Two secondary/complementary checks remain useful:

  1. recall_at_k against a small hand-curated set of problem groups -- kept
     around for quick, dependency-free sanity checks on a handful of
     obviously-related families (e.g. Two Sum / 3Sum / 4Sum) while iterating.

  2. Visual inspection via seaborn clustermap on a small subset (the curated
     groups plus a random sample) -- full pairwise clustermaps don't render
     usefully at 2600+ problems, so this keeps the subset small enough to
     read the tags alongside the dendrogram by eye.

Usage:
    import categorical_similarity as cs
    import evaluate_categorical_similarity as ev

    df = cs.load_dataset()
    tag_sim = cs.tag_similarity_matrix(df, idf_weighted=True)
    sig_sim = cs.signature_similarity_matrix(df)
    combined = cs.combine(tag_sim, sig_sim)
    variants = {"tags": tag_sim, "signature": sig_sim, "combined": combined}

    ground_truth = ev.load_similar_questions_ground_truth(df)
    ev.evaluate_against_ground_truth(variants, ground_truth, k=10)
    ev.k_sweep(variants, ground_truth, k_values=[1, 3, 5, 10, 20])

    ev.recall_at_k(variants, k=5)          # secondary spot-check
    ev.plot_clustermap(combined, df)
"""

import ast
import os
import pickle

import numpy as np
import pandas as pd

GROUND_TRUTH_CACHE_PATH = os.path.join(os.path.dirname(__file__), "similar_questions_cache.pkl")

# Hand-picked groups of problems that share an obvious technique/pattern.
# Every name below was checked to exist in LeetCodeDataset-train.
CURATED_GROUPS = [
    ["two-sum", "3sum", "3sum-closest", "4sum"],
    ["climbing-stairs", "fibonacci-number", "min-cost-climbing-stairs"],
    ["merge-two-sorted-lists", "merge-k-sorted-lists"],
    [
        "binary-tree-inorder-traversal",
        "binary-tree-preorder-traversal",
        "binary-tree-postorder-traversal",
    ],
    ["valid-parentheses", "generate-parentheses"],
    ["maximum-subarray", "maximum-product-subarray"],
    ["number-of-islands", "max-area-of-island", "surrounded-regions"],
    ["reverse-linked-list", "reverse-linked-list-ii"],
    ["best-time-to-buy-and-sell-stock", "best-time-to-buy-and-sell-stock-ii"],
    ["coin-change", "coin-change-ii"],
    ["word-search", "word-search-ii"],
    ["house-robber", "house-robber-ii", "house-robber-iii"],
]


def load_similar_questions_ground_truth(df: pd.DataFrame, refresh: bool = False) -> dict[str, list[str]]:
    """
    Loads LeetCode's own "Similar Questions" edges (scraped in the
    kaysss/leetcode-problem-detailed HF dataset, keyed by the same slug
    `categorical_similarity.load_dataset` uses as "name") and restricts them
    to problems present in `df` on both ends -- an edge pointing outside our
    working set can't be retrieved from our own candidate pool, so it's
    dropped rather than counted as an unreachable miss.

    Returns {name: [relevant_name, ...]} for every problem in `df` that has
    at least one such edge landing back inside `df` (~68% of the train split
    have any edge at all; a handful more get dropped here because their only
    listed neighbors fall outside the split).
    """
    if not refresh and os.path.exists(GROUND_TRUTH_CACHE_PATH):
        raw_map = pd.read_pickle(GROUND_TRUTH_CACHE_PATH)
    else:
        sq_df = pd.read_csv("hf://datasets/kaysss/leetcode-problem-detailed/questions_detailed.csv")
        sq_df["similar_list"] = sq_df["similarQuestions"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        raw_map = dict(zip(sq_df["TitleSlug"], sq_df["similar_list"]))
        pd.to_pickle(raw_map, GROUND_TRUTH_CACHE_PATH)

    valid_names = set(df["name"])
    ground_truth = {}
    for name in valid_names:
        # A handful of LeetCode's own scraped edges list a problem as its own
        # "similar question" (e.g. jump-game-vii, stone-game-ix) -- 4 out of
        # 4052 edges. Harmless to leave in (a query can never retrieve itself
        # anyway, since evaluate_against_ground_truth drops self from the
        # candidate side), but it inflates that query's relevant-set size for
        # no reason, so drop it here instead.
        kept = [n for n in raw_map.get(name, []) if n in valid_names and n != name]
        if kept:
            ground_truth[name] = kept
    return ground_truth


LEGITIMACY_CACHE_PATH = os.path.join(os.path.dirname(__file__), "ground_truth_legitimacy_cache.pkl")

ROMAN_NUMERALS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}


def name_family_base(name: str) -> str:
    """Strips a trailing roman-numeral 'sequel' token (and anything after it)
    from a problem slug, e.g. 'two-sum-ii-input-array-is-sorted' -> 'two-sum',
    'jump-game-vii' -> 'jump-game'. Used to auto-flag same-title-family edges
    as narrative-driven rather than algorithm-driven -- every sampled instance
    of this category turned out spurious on manual inspection (e.g.
    jump-game-vi/vii share nothing but the title; see conversation history)."""
    tokens = name.split("-")
    for i, tok in enumerate(tokens):
        if tok in ROMAN_NUMERALS:
            return "-".join(tokens[:i])
    return name


def is_name_family_pair(a: str, b: str) -> bool:
    """True if `a` and `b` reduce to the same title-family base and at least
    one of them actually had a roman-numeral suffix stripped (i.e. one is a
    literal "sequel" of the other, not just two unrelated names that happen to
    already match). Order-independent -- unlike checking name_family_base(a)
    != a alone, which silently misses a pair where the *un-suffixed* name
    happens to be sorted first (e.g. ("erect-the-fence",
    "erect-the-fence-ii"))."""
    base_a, base_b = name_family_base(a), name_family_base(b)
    return base_a == base_b and (base_a != a or base_b != b)


def load_ground_truth_legitimacy() -> dict[frozenset, str]:
    """
    Loads the hand-judged legitimacy cache: {frozenset({a, b}): label} where
    label is one of "LEGITIMATE" (real shared technique), "LOOSE" (defensible
    conceptual/family connection but a different specific technique), or
    "SPURIOUS" / "SPURIOUS_NAME_FAMILY" (no real technique connection --
    narrative/name-family or superficial tag-only overlap).

    This exists because LeetCode's own "Similar Questions" list is not purely
    algorithm-driven -- manual review of all 188 ground-truth pairs reachable
    in our classified pool (as of this cache) found 31.4% spurious (21.8%
    superficial + 9.6% name-family), meaning roughly a third of "misses"
    against raw ground truth are the metric penalizing correct behavior, not
    real failures. See `filter_ground_truth_by_legitimacy` to use this for a
    more trustworthy eval. The cache only covers pairs judged so far -- as
    the classified pool grows, new pairs need judging (or will pass through
    unfiltered under the default policy; see that function).
    """
    if not os.path.exists(LEGITIMACY_CACHE_PATH):
        return {}
    with open(LEGITIMACY_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def filter_ground_truth_by_legitimacy(
    ground_truth: dict[str, list[str]],
    legitimacy: dict[frozenset, str] | None = None,
    keep_labels: set[str] = frozenset({"LEGITIMATE", "LOOSE"}),
    unjudged_policy: str = "keep",
) -> dict[str, list[str]]:
    """
    Filters a ground-truth dict down to edges whose hand-judged legitimacy
    label is in `keep_labels` (default: keep LEGITIMATE and LOOSE, drop both
    SPURIOUS categories). `unjudged_policy` controls what happens to a pair
    that was never manually judged (e.g. because the classified pool has
    grown since the legitimacy cache was built):
      "keep"   -- pass it through unfiltered (conservative; won't silently
                  drop a real edge just because nobody's reviewed it yet).
      "drop"   -- exclude it (strict; only ever scores on vetted edges).
    """
    if legitimacy is None:
        legitimacy = load_ground_truth_legitimacy()
    if unjudged_policy not in ("keep", "drop"):
        raise ValueError(f"Unknown unjudged_policy: {unjudged_policy!r}")

    filtered = {}
    for query, relevant in ground_truth.items():
        kept = []
        for target in relevant:
            label = legitimacy.get(frozenset((query, target)))
            if label is None:
                if unjudged_policy == "keep":
                    kept.append(target)
            elif label in keep_labels:
                kept.append(target)
        if kept:
            filtered[query] = kept
    return filtered


def ground_truth_seed_sample(df: pd.DataFrame, ground_truth: dict[str, list[str]], target_new: int, already_cached: set[str] = frozenset(), seed: int = 1) -> set[str]:
    """
    Samples problems for an LLM-labeling pass by growing "seed + its actual
    ground-truth neighbors" groups instead of sampling by tag -- since a
    query is only evaluable if a real relevant neighbor also ends up
    classified, this guarantees every seed becomes evaluable by construction,
    rather than hoping tag-stratified sampling incidentally overlaps.

    Motivation: the ground-truth graph turned out to be one giant connected
    component (1276 of 1738 ground-truth-linked problems) plus a long tail of
    small ones -- "include whole connected components" doesn't work (the
    giant one is most of the graph), but seed+neighbor expansion sidesteps
    that entirely, since it only ever pulls in each seed's direct neighbors,
    not its whole component. Empirically ~5-6x more evaluable queries per
    classification than tag-stratified sampling (e.g. ~100 new classifications
    -> ~138 evaluable queries, vs 182 tag-stratified -> only 24).

    `already_cached` lets already-classified problems count for free -- seeds
    already in it are skipped, and their neighbors don't count against
    `target_new`. Returns the full sample (already_cached union newly-needed
    problems), not just the new ones -- pass this straight to a discovery
    batch runner, which will skip anything already cached.
    """
    seed_order = cs_stratified_order(df, seed=seed)
    seed_candidates = [n for n in seed_order if n in ground_truth]

    sample = set(already_cached)
    for name in seed_candidates:
        if len(sample - already_cached) >= target_new:
            break
        if name in sample:
            continue
        sample |= {name} | set(ground_truth[name])
    return sample


def cs_stratified_order(df: pd.DataFrame, seed: int) -> list[str]:
    """Diverse (tag-stratified) ordering of problem names, reusing
    categorical_similarity's rarest-tag-first sampling with a large enough
    quota to just produce an ordering rather than a small subset."""
    import categorical_similarity as cs

    ordered_df = cs.stratified_sample_by_tag(df, per_tag_quota=len(df), seed=seed)
    return ordered_df["name"].tolist()


def evaluate_against_ground_truth(sim_variants: dict[str, pd.DataFrame], ground_truth: dict[str, list[str]], k: int = 10) -> pd.DataFrame:
    """
    For every query problem in `ground_truth`, checks its top-k neighbors
    under each similarity variant against that problem's real relevant set
    and reports three averaged metrics per variant:

      hit_rate@k    -- fraction of queries where >=1 relevant problem appears
                       in the top k (comparable to the old curated recall_at_k).
      recall@k      -- MACRO-average: for each query, its own hit-fraction
                       (|hits|/|its relevant set|), then averaged across
                       queries with every query weighted equally regardless
                       of how many ground-truth edges it has. A query with 15
                       listed relevant problems (e.g. two-sum) counts the same
                       as one with 1 -- this answers "how well do we do on a
                       typical query," not "what fraction of all edges did we
                       recover."
      pooled_recall@k -- MICRO-average: total hits summed across every query,
                       divided by the total size of every query's relevant set
                       summed together. This is the number that directly
                       answers "of all the ground-truth edges we have, how
                       many did we actually retrieve" -- queries with larger
                       relevant sets contribute proportionally more, since
                       they contribute more edges to both the numerator and
                       denominator.
    """
    rows = []
    for variant_name, sim in sim_variants.items():
        hit_flags, recalls = [], []
        total_hits, total_relevant = 0, 0
        skipped = 0
        for name, relevant in ground_truth.items():
            if name not in sim.index:
                skipped += 1
                continue
            relevant_set = set(relevant)
            neighbors = set(sim[name].drop(index=name).sort_values(ascending=False).head(k).index)
            hits = neighbors & relevant_set
            hit_flags.append(1.0 if hits else 0.0)
            recalls.append(len(hits) / len(relevant_set))
            total_hits += len(hits)
            total_relevant += len(relevant_set)
        if skipped:
            print(f"[evaluate_against_ground_truth] '{variant_name}': {skipped} query problems not found in similarity matrix, skipped")
        rows.append(
            {
                "variant": variant_name,
                "n_queries": len(hit_flags),
                f"hit_rate@{k}": np.mean(hit_flags) if hit_flags else np.nan,
                f"recall@{k}": np.mean(recalls) if recalls else np.nan,
                f"pooled_recall@{k}": total_hits / total_relevant if total_relevant else np.nan,
            }
        )
    return pd.DataFrame(rows)


def k_sweep(sim_variants: dict[str, pd.DataFrame], ground_truth: dict[str, list[str]], k_values: list[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
    """Runs evaluate_against_ground_truth at each k and stacks the results,
    so you can see how quickly each variant's recall improves as k grows --
    useful for picking how many candidates a future recommender should
    actually surface."""
    frames = []
    for k in k_values:
        result = evaluate_against_ground_truth(sim_variants, ground_truth, k=k)
        result = result.rename(columns={f"hit_rate@{k}": "hit_rate", f"recall@{k}": "recall", f"pooled_recall@{k}": "pooled_recall"})
        result["k"] = k
        frames.append(result)
    return pd.concat(frames, ignore_index=True)[["variant", "k", "n_queries", "hit_rate", "recall", "pooled_recall"]]


def recall_at_k(sim_variants: dict[str, pd.DataFrame], k: int = 5, groups: list[list[str]] = CURATED_GROUPS) -> pd.DataFrame:
    """
    For each curated group and each named similarity matrix in
    `sim_variants`, checks -- for every problem in the group -- whether its
    top-k neighbors (by that similarity matrix) include at least one other
    member of the same group. Reports per-group hit rate and the overall
    average, one column per variant, so variants are directly comparable.
    """
    rows = []
    for group in groups:
        row = {"group": ", ".join(group), "size": len(group)}
        group_set = set(group)
        for variant_name, sim in sim_variants.items():
            available = [p for p in group if p in sim.index]
            missing = group_set - set(available)
            if missing:
                print(f"[recall_at_k] '{variant_name}': group members not found, skipping them: {missing}")
            hits = 0
            for p in available:
                neighbors = set(sim[p].drop(index=p).sort_values(ascending=False).head(k).index)
                if neighbors & (group_set - {p}):
                    hits += 1
            row[variant_name] = hits / len(available) if available else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    variant_cols = list(sim_variants.keys())
    avg_row = {"group": "AVERAGE", "size": result["size"].sum()}
    avg_row.update({c: result[c].mean() for c in variant_cols})
    result = pd.concat([result, pd.DataFrame([avg_row])], ignore_index=True)
    return result


def plot_clustermap(sim: pd.DataFrame, df: pd.DataFrame, subset: list[str] | None = None, n_random: int = 20, seed: int = 0, figsize=(16, 16)):
    """
    Clustermap of `sim` restricted to a readable subset: every problem named
    in the curated groups (so related problems are there to check against
    each other) plus `n_random` randomly sampled problems for contrast.
    Row/column labels are annotated with a truncated tag list so you can
    visually sanity-check whether nearby clusters actually share topics.
    """
    import seaborn as sns

    if subset is None:
        curated_names = [p for group in CURATED_GROUPS for p in group if p in sim.index]
        rng = np.random.default_rng(seed)
        pool = [n for n in sim.index if n not in curated_names]
        random_names = rng.choice(pool, size=min(n_random, len(pool)), replace=False).tolist()
        subset = curated_names + random_names

    sub_sim = sim.loc[subset, subset]
    tags_by_name = df.set_index("name")["tags"]

    def label(name: str) -> str:
        tags = ", ".join(tags_by_name.get(name, [])[:3])
        return f"{name} [{tags}]"

    labels = [label(n) for n in subset]

    sns.set(font_scale=0.7)
    return sns.clustermap(
        sub_sim,
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlGnBu",
        figsize=figsize,
    )
