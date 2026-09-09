"""
Empirically checks whether the gated-corroboration bonus (structural_
corroboration.apply_gated_corroboration) actually improves retrieval on top
of the current best checkpoint (combined+mechanism, alpha=0.75), rather than
assuming the "richness-gated" design helps just because the raw-agreement
numbers in scratch_structural_validation.py looked directionally right.

Sweeps bonus_weight at a fixed, already-validated richness threshold (8,
from scratch_structural_validation.py) against the FILTERED ground truth
(legitimacy-judged LEGITIMATE+LOOSE edges only), since that's the more
trustworthy signal established earlier in the project.

Usage:
    python run_corroboration_eval.py
"""

import pickle

import numpy as np
import pandas as pd

import categorical_similarity as cs
import evaluate_categorical_similarity as ev
import structural_corroboration as sc

ABSTRACT_CACHE_PATH = "abstract_discovery_cache.pkl"
MECHANISM_EMBEDDING_CACHE_PATH = "mechanism_embedding_cache.pkl"


def main():
    df = cs.load_dataset()

    with open(ABSTRACT_CACHE_PATH, "rb") as f:
        abstracts = pickle.load(f)
    with open(MECHANISM_EMBEDDING_CACHE_PATH, "rb") as f:
        mechanism_embeddings = pickle.load(f)
    with open("/tmp/structural_tags.pkl", "rb") as f:
        structural_tags = pickle.load(f)
    structural_sim = pd.read_pickle("/tmp/structural_similarity_matrix.pkl")

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

    base = 0.75 * mechanism_sim + 0.25 * combined
    structural_sim = structural_sim.reindex(index=names, columns=names).fillna(0.0)

    variants = {"base (combined+mechanism, alpha=0.75)": base}
    for threshold in [6, 8, 10]:
        for bonus_weight in [0.05, 0.1, 0.2, 0.3]:
            key = f"+corroboration (thresh={threshold}, w={bonus_weight})"
            variants[key] = sc.apply_gated_corroboration(base, structural_sim, structural_tags, bonus_weight, threshold)

    # ungated ablation: what if we add the SAME bonus with no richness gate
    # at all (every pair, however sparse, contributes)? this isolates
    # whether the gate itself is doing anything, vs. the bonus alone.
    for bonus_weight in [0.05, 0.1, 0.2]:
        variants[f"+corroboration UNGATED (w={bonus_weight})"] = base + bonus_weight * structural_sim

    ground_truth_full = ev.load_similar_questions_ground_truth(df)
    ground_truth = {name: [n for n in rel if n in pool] for name, rel in ground_truth_full.items() if name in pool}
    ground_truth = {name: rel for name, rel in ground_truth.items() if rel}

    legitimacy = ev.load_ground_truth_legitimacy()
    ground_truth_filtered = ev.filter_ground_truth_by_legitimacy(ground_truth, legitimacy)
    print(f"Evaluable queries (legitimacy-filtered): {len(ground_truth_filtered)}")

    print("\n=== FILTERED ground truth (SPURIOUS + SPURIOUS_NAME_FAMILY dropped) ===")
    print(ev.evaluate_against_ground_truth(variants, ground_truth_filtered, k=10).to_string(index=False))


if __name__ == "__main__":
    main()
