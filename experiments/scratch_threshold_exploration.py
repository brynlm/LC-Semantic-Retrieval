"""
Explores whether a similarity-score threshold can separate LEGITIMATE
matches from LOOSE/SPURIOUS ones and from the general background of
unrelated pairs -- the prerequisite for a future "confidence threshold"
feature (return few/no recommendations for a query with no strong match,
let the user manually lower the bar to see weaker candidates instead of
always forcing k results).

Reports score-distribution percentiles per ground-truth legitimacy label,
plus a random-pair background distribution (proxy for "what a query with no
declared relationship at all looks like"), then sweeps candidate thresholds
to see how much of each group survives at each cutoff.

Usage:
    python scratch_threshold_exploration.py
"""

import pickle
import random

import numpy as np
import pandas as pd

import categorical_similarity as cs
import evaluate_categorical_similarity as ev

ALPHA = 0.75


def main():
    df = cs.load_dataset()
    with open("abstract_discovery_cache.pkl", "rb") as f:
        abstracts = pickle.load(f)
    with open("mechanism_embedding_cache.pkl", "rb") as f:
        mechanism_embeddings = pickle.load(f)

    pool = sorted(set(abstracts) & set(mechanism_embeddings) & set(df["name"]))
    pool_df = df[df["name"].isin(pool)].reset_index(drop=True)
    tag_sim = cs.tag_similarity_matrix(pool_df, idf_weighted=True)
    sig_sim = cs.signature_similarity_matrix(pool_df)
    combined = cs.combine(tag_sim, sig_sim)
    names = list(pool_df["name"])
    vectors = np.stack([mechanism_embeddings[n] for n in names])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    mechanism_sim = pd.DataFrame(vectors @ vectors.T, index=names, columns=names)
    base = ALPHA * mechanism_sim + (1 - ALPHA) * combined

    legitimacy = ev.load_ground_truth_legitimacy()

    scores_by_label = {"LEGITIMATE": [], "LOOSE": [], "SPURIOUS": [], "SPURIOUS_NAME_FAMILY": []}
    labeled_pairs = set()
    for pair, label in legitimacy.items():
        a, b = tuple(pair)
        if a not in pool or b not in pool:
            continue
        s = base.loc[a, b]
        scores_by_label.setdefault(label, []).append(s)
        labeled_pairs.add(frozenset((a, b)))

    random.seed(0)
    background_scores = []
    while len(background_scores) < 5000:
        a, b = random.sample(names, 2)
        if frozenset((a, b)) in labeled_pairs:
            continue
        background_scores.append(base.loc[a, b])

    print(f"=== Score distributions (combined+mechanism, alpha={ALPHA}) ===")
    for label, scores in scores_by_label.items():
        if not scores:
            continue
        arr = np.array(scores)
        print(f"{label:22s} n={len(arr):4d}  mean={arr.mean():.3f}  p25={np.percentile(arr,25):.3f}  "
              f"median={np.median(arr):.3f}  p75={np.percentile(arr,75):.3f}  p10={np.percentile(arr,10):.3f}")
    arr = np.array(background_scores)
    print(f"{'RANDOM (no declared edge)':22s} n={len(arr):4d}  mean={arr.mean():.3f}  p25={np.percentile(arr,25):.3f}  "
          f"median={np.median(arr):.3f}  p75={np.percentile(arr,75):.3f}  p10={np.percentile(arr,10):.3f}")

    print(f"\n=== Threshold sweep: fraction of each group with score >= t ===")
    print(f"{'t':>6s} {'LEGITIMATE':>11s} {'LOOSE':>8s} {'SPURIOUS':>9s} {'SPUR_NAME':>10s} {'RANDOM':>8s}")
    for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        row = [t]
        for label in ["LEGITIMATE", "LOOSE", "SPURIOUS", "SPURIOUS_NAME_FAMILY"]:
            arr = np.array(scores_by_label.get(label, []))
            row.append(np.mean(arr >= t) if len(arr) else float("nan"))
        row.append(np.mean(np.array(background_scores) >= t))
        print(f"{row[0]:6.2f} {row[1]:11.3f} {row[2]:8.3f} {row[3]:9.3f} {row[4]:10.3f} {row[5]:8.3f}")


if __name__ == "__main__":
    main()
