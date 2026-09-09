"""
Re-runs the ground-truth evaluation with the hand-judged legitimacy filter
applied, to see how much of the "combined + mechanism" win survives once
narrative/spurious ground-truth edges are excluded rather than penalizing
against them.

Restricted to the classified pool (abstract_discovery_cache /
mechanism_embedding_cache) since that's the only subset with mechanism
abstracts/embeddings computed. Sweeps the categorical/mechanism blend
weight so the comparison isn't dependent on one arbitrarily-picked alpha.

Usage:
    python run_filtered_eval.py
"""

import pickle

import numpy as np
import pandas as pd

import categorical_similarity as cs
import evaluate_categorical_similarity as ev

ABSTRACT_CACHE_PATH = "abstract_discovery_cache.pkl"
MECHANISM_EMBEDDING_CACHE_PATH = "mechanism_embedding_cache.pkl"


def main():
    df = cs.load_dataset()

    with open(ABSTRACT_CACHE_PATH, "rb") as f:
        abstracts = pickle.load(f)
    with open(MECHANISM_EMBEDDING_CACHE_PATH, "rb") as f:
        mechanism_embeddings = pickle.load(f)

    pool = sorted(set(abstracts) & set(mechanism_embeddings) & set(df["name"]))
    print(f"Classified pool: {len(pool)} problems")
    pool_df = df[df["name"].isin(pool)].reset_index(drop=True)

    tag_sim = cs.tag_similarity_matrix(pool_df, idf_weighted=True)
    sig_sim = cs.signature_similarity_matrix(pool_df)
    combined = cs.combine(tag_sim, sig_sim)

    names = list(pool_df["name"])
    vectors = np.stack([mechanism_embeddings[n] for n in names])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cos_sim = vectors @ vectors.T
    mechanism_sim = pd.DataFrame(cos_sim, index=names, columns=names)

    variants = {"categorical (tags+sig)": combined, "mechanism_alone": mechanism_sim}
    for alpha in [0.25, 0.5, 0.75]:
        blend = alpha * mechanism_sim + (1 - alpha) * combined
        variants[f"combined+mechanism (alpha={alpha})"] = blend

    ground_truth_full = ev.load_similar_questions_ground_truth(df)
    ground_truth = {name: [n for n in rel if n in pool] for name, rel in ground_truth_full.items() if name in pool}
    ground_truth = {name: rel for name, rel in ground_truth.items() if rel}
    print(f"Evaluable queries (unfiltered): {len(ground_truth)}")

    legitimacy = ev.load_ground_truth_legitimacy()
    ground_truth_filtered = ev.filter_ground_truth_by_legitimacy(ground_truth, legitimacy)
    # unjudged_policy="drop" here (unlike the LEGITIMATE+LOOSE view above,
    # which keeps unjudged edges for continuity with earlier sessions): the
    # legitimacy cache was hand-labeled against an earlier, smaller pool, and
    # as the pool grows (run_pool_expansion.py) most newly-reachable edges
    # are unjudged -- silently keeping them here would dilute "LEGITIMATE
    # only, strictest" with unvetted pairs, defeating its purpose. This
    # means growing the pool naturally shrinks this view's query count until
    # someone runs a new legitimacy-labeling pass over the new edges.
    ground_truth_legit_only = ev.filter_ground_truth_by_legitimacy(ground_truth, legitimacy, keep_labels={"LEGITIMATE"}, unjudged_policy="drop")
    print(f"Evaluable queries (legitimacy-filtered, LEGITIMATE+LOOSE): {len(ground_truth_filtered)}")
    print(f"Evaluable queries (LEGITIMATE only, vetted edges): {len(ground_truth_legit_only)}")

    print("\n=== UNFILTERED ground truth (includes spurious/narrative edges) ===")
    print(ev.evaluate_against_ground_truth(variants, ground_truth, k=10).to_string(index=False))

    print("\n=== FILTERED ground truth (SPURIOUS + SPURIOUS_NAME_FAMILY dropped, LEGITIMATE+LOOSE kept) ===")
    print(ev.evaluate_against_ground_truth(variants, ground_truth_filtered, k=10).to_string(index=False))

    # LOOSE edges are a defensible conceptual/family link, not necessarily the
    # same distinguishing technique -- diagnosis (scratch_miss_diagnosis.py)
    # found these are systematically ~30pp harder to hit than LEGITIMATE
    # edges, dragging the blended number below what "does it find the real
    # matching technique" actually looks like. Report LEGITIMATE-only
    # alongside the blended number as the more honest core-capability read.
    print("\n=== LEGITIMATE-only ground truth (strictest: same distinguishing technique) ===")
    print(ev.evaluate_against_ground_truth(variants, ground_truth_legit_only, k=10).to_string(index=False))


if __name__ == "__main__":
    main()
