"""
Unsupervised clustering + tag-purity analysis: an evaluation approach that
doesn't depend on LeetCode's sparse "Similar Questions" ground truth (which
can't tell us anything about a genuinely-good match that LeetCode just never
happened to cross-list) or on manual pair-by-pair audits (which don't scale
to the whole embedding space). Instead: cluster the embeddings directly, then
check whether the resulting clusters are more tag-coherent than chance, using
the already-validated IDF-weighted tag-similarity machinery from
categorical_similarity.py (rare shared tags count for more than common ones,
same logic as the rest of this project's tag-similarity work).

Caveat (important, not hidden): tags are coarse. A cluster can have perfect
tag purity while still mixing genuinely-different specific techniques under
one broad tag like "Dynamic Programming" -- this is exactly the limitation
that motivated moving past tag-based similarity to mechanism abstracts in
this project. High purity is reassuring; it is not proof of fine-grained
correctness. Best read as a signal for catching gross clustering failures,
supplemented by direct spot-checks of specific clusters.

Usage:
    python scratch_cluster_purity.py
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import categorical_similarity as cs

N_CLUSTERS_OPTIONS = [20, 30, 40]


def build_embedding_matrix(cache, names):
    vecs = np.stack([cache[n] for n in names])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def cluster_purity_report(label, emb_matrix, names, tag_sim, k, random_baseline):
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(emb_matrix)

    cluster_purities = []
    cluster_sizes = []
    for c in range(k):
        members = [names[i] for i in range(len(names)) if labels[i] == c]
        cluster_sizes.append(len(members))
        if len(members) < 2:
            continue
        sims = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sims.append(tag_sim.loc[members[i], members[j]])
        cluster_purities.append((len(members), float(np.mean(sims)), members))

    weighted_purity = sum(n * p for n, p, _ in cluster_purities) / sum(n for n, p, _ in cluster_purities)
    print(f"{label} (k={k}): weighted mean intra-cluster tag-similarity = {weighted_purity:.3f} "
          f"(random baseline = {random_baseline:.3f}), cluster sizes: min={min(cluster_sizes)} "
          f"max={max(cluster_sizes)} mean={np.mean(cluster_sizes):.1f}")
    return labels, cluster_purities, weighted_purity


def main():
    df = cs.load_dataset()
    with open("mechanism_embedding_cache.pkl", "rb") as f:
        prod_emb = pickle.load(f)
    with open("merged_code_only_mechanism_embedding_cache.pkl", "rb") as f:
        code_emb = pickle.load(f)

    names = sorted(code_emb.keys())
    pool_df = df[df["name"].isin(names)].reset_index(drop=True)
    print(f"Pool size: {len(names)}")

    tag_sim = cs.tag_similarity_matrix(pool_df, idf_weighted=True, metric="dice")
    tag_sim.to_pickle("/tmp/cluster_tag_sim.pkl")

    # random baseline: mean tag-similarity over random same-size groups, for comparison
    rng = np.random.default_rng(0)
    baseline_vals = []
    for _ in range(2000):
        a, b = rng.choice(names, 2, replace=False)
        baseline_vals.append(tag_sim.loc[a, b])
    random_baseline = float(np.mean(baseline_vals))
    print(f"Random-pair tag-similarity baseline: {random_baseline:.3f}")
    print()

    prod_matrix = build_embedding_matrix(prod_emb, names)
    code_matrix = build_embedding_matrix(code_emb, names)

    results = {}
    for k in N_CLUSTERS_OPTIONS:
        print(f"--- k={k} ---")
        prod_labels, prod_purities, prod_wp = cluster_purity_report("PRODUCTION", prod_matrix, names, tag_sim, k, random_baseline)
        code_labels, code_purities, code_wp = cluster_purity_report("CODE-ONLY ", code_matrix, names, tag_sim, k, random_baseline)
        results[k] = {
            "production": {"weighted_purity": prod_wp, "labels": prod_labels.tolist(), "cluster_purities": [(n, p) for n, p, _ in prod_purities]},
            "code_only": {"weighted_purity": code_wp, "labels": code_labels.tolist(), "cluster_purities": [(n, p) for n, p, _ in code_purities]},
        }
        print()

    with open("/tmp/cluster_results.pkl", "wb") as f:
        pickle.dump({"names": names, "results": results, "random_baseline": random_baseline}, f)

    # save full cluster membership for k=30 (middle option) for spot-checking
    k = 30
    km_prod = KMeans(n_clusters=k, n_init=10, random_state=0).fit(prod_matrix)
    km_code = KMeans(n_clusters=k, n_init=10, random_state=0).fit(code_matrix)
    with open("/tmp/cluster_membership_k30.pkl", "wb") as f:
        pickle.dump({
            "names": names,
            "production_labels": km_prod.labels_.tolist(),
            "code_only_labels": km_code.labels_.tolist(),
        }, f)
    print("Saved cluster membership for k=30 for spot-checking.")


if __name__ == "__main__":
    main()
