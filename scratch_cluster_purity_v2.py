"""
Rerun of scratch_cluster_purity.py's purity comparison using freshly-scraped
LeetCode tags (scratch_refresh_tags.py) instead of the stale local snapshot.
Two systematic renames (Graph->Graph Theory, Union Find->Union-Find) and 304
genuinely new, finer-grained tags (Minimax, Knapsack Problem, Dijkstra's
Algorithm, DP on Trees, ...) were found missing from the local data across
515/2641 problems -- exactly the kind of specific-technique granularity that
could change whether a cluster like code-only's cluster 6 (Hash Table-driven,
purity=0.445, flagged as likely "shared substep not shared technique") reads
as genuinely coherent or not.

Cluster MEMBERSHIP is unchanged (it depends only on embeddings, not tags), so
this reuses the saved labels from scratch_cluster_purity.py's run rather than
re-clustering -- only the tag_sim matrix (and therefore purity) is recomputed.

Usage:
    python scratch_cluster_purity_v2.py
"""

import pickle

import numpy as np

import categorical_similarity as cs

with open("/tmp/tag_refresh_fresh_tags.pkl", "rb") as f:
    fresh_tags = pickle.load(f)

with open("/tmp/cluster_results.pkl", "rb") as f:
    cr = pickle.load(f)
names = cr["names"]

df = cs.load_dataset()
pool_df = df[df["name"].isin(names)].reset_index(drop=True)
assert len(pool_df) == len(names)

# swap in fresh tags for this pool
pool_df["tags"] = pool_df["name"].map(fresh_tags)
assert pool_df["tags"].isna().sum() == 0

tag_sim = cs.tag_similarity_matrix(pool_df, idf_weighted=True, metric="dice")
tag_sim.to_pickle("/tmp/cluster_tag_sim_v2.pkl")

rng = np.random.default_rng(0)
baseline_vals = [tag_sim.loc[a, b] for a, b in
                 (rng.choice(names, 2, replace=False) for _ in range(2000))]
random_baseline = float(np.mean(baseline_vals))
print(f"Random-pair tag-similarity baseline (fresh tags): {random_baseline:.3f}")
print()


def weighted_purity_for(labels, k):
    cluster_purities = []
    for c in range(k):
        members = [names[i] for i in range(len(names)) if labels[i] == c]
        if len(members) < 2:
            continue
        sims = [tag_sim.loc[members[i], members[j]]
                for i in range(len(members)) for j in range(i + 1, len(members))]
        cluster_purities.append((len(members), float(np.mean(sims))))
    wp = sum(n * p for n, p in cluster_purities) / sum(n for n, p in cluster_purities)
    return wp, cluster_purities


results_v2 = {}
for k, res in cr["results"].items():
    prod_wp, prod_cp = weighted_purity_for(res["production"]["labels"], k)
    code_wp, code_cp = weighted_purity_for(res["code_only"]["labels"], k)
    print(f"k={k}: production old={res['production']['weighted_purity']:.3f} new={prod_wp:.3f} | "
          f"code-only old={res['code_only']['weighted_purity']:.3f} new={code_wp:.3f}")
    results_v2[k] = {
        "production": {"weighted_purity": prod_wp, "cluster_purities": prod_cp, "labels": res["production"]["labels"]},
        "code_only": {"weighted_purity": code_wp, "cluster_purities": code_cp, "labels": res["code_only"]["labels"]},
    }

with open("/tmp/cluster_results_v2.pkl", "wb") as f:
    pickle.dump({"names": names, "results": results_v2, "random_baseline": random_baseline}, f)

# specifically re-check the flagged code-only cluster 6 at k=30
with open("/tmp/cluster_membership_k30.pkl", "rb") as f:
    memb = pickle.load(f)
code_labels_30 = memb["code_only_labels"]
cluster6_members = [names[i] for i in range(len(names)) if code_labels_30[i] == 6]
print()
print(f"code-only cluster 6 (k=30), n={len(cluster6_members)}:")
sims = [tag_sim.loc[cluster6_members[i], cluster6_members[j]]
        for i in range(len(cluster6_members)) for j in range(i + 1, len(cluster6_members))]
print(f"  new purity = {np.mean(sims):.3f} (was 0.445 with stale tags)")
print("  fresh tags per member:")
for m in cluster6_members:
    print(f"    {m}: {fresh_tags[m]}")
