import pickle
import numpy as np
import pandas as pd
import categorical_similarity as cs
import evaluate_categorical_similarity as ev

df = cs.load_dataset()
with open("abstract_discovery_cache.pkl", "rb") as f:
    abstracts = pickle.load(f)
with open("mechanism_embedding_cache.pkl", "rb") as f:
    mech_emb = pickle.load(f)

pool = sorted(set(abstracts) & set(mech_emb) & set(df["name"]))
pool_df = df[df["name"].isin(pool)].reset_index(drop=True)

tag_sim = cs.tag_similarity_matrix(pool_df, idf_weighted=True)
sig_sim = cs.signature_similarity_matrix(pool_df)
combined = cs.combine(tag_sim, sig_sim)

names = list(pool_df["name"])
vectors = np.stack([mech_emb[n] for n in names])
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
cos_sim = vectors @ vectors.T
mechanism_sim = pd.DataFrame(cos_sim, index=names, columns=names)

alpha = 0.75
blend = alpha * mechanism_sim + (1 - alpha) * combined

ground_truth_full = ev.load_similar_questions_ground_truth(df)
ground_truth = {name: [n for n in rel if n in pool] for name, rel in ground_truth_full.items() if name in pool}
ground_truth = {name: rel for name, rel in ground_truth.items() if rel}
legitimacy = ev.load_ground_truth_legitimacy()
gt_filtered = ev.filter_ground_truth_by_legitimacy(ground_truth, legitimacy)

k = 10
hits, near_misses, deep_misses = [], [], []
for name, relevant in gt_filtered.items():
    if name not in blend.index:
        continue
    relevant_set = set(relevant)
    ranked = blend[name].drop(index=name).sort_values(ascending=False)
    top_k = set(ranked.head(k).index)
    hit = top_k & relevant_set
    if hit:
        hits.append(name)
        continue
    top20 = set(ranked.head(20).index)
    if top20 & relevant_set:
        near_misses.append((name, relevant))
    else:
        deep_misses.append((name, relevant, ranked))

print(f"hits={len(hits)} near_misses={len(near_misses)} deep_misses={len(deep_misses)} total={len(gt_filtered)}")
print()
print("=== DEEP MISSES ===")
for name, relevant, ranked in deep_misses:
    print(f"{name}  (relevant: {relevant})")
    for r in relevant:
        rank = list(ranked.index).index(r) + 1 if r in ranked.index else "N/A"
        score = ranked.get(r, float("nan"))
        label = legitimacy.get(frozenset((name, r)), "?")
        print(f"    -> {r}: rank={rank} score={score:.3f} [{label}]")
print()
print("=== NEAR MISSES (partner in rank 11-20) ===")
for name, relevant in near_misses:
    print(f"{name}  (relevant: {relevant})")
