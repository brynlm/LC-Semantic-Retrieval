import pickle
import numpy as np
import pandas as pd
import categorical_similarity as cs
import evaluate_categorical_similarity as ev

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
deep_misses = []
near_misses = []
hits = []
for name, relevant in gt_filtered.items():
    if name not in blend.index:
        continue
    relevant_set = set(relevant)
    ranked = blend[name].drop(index=name).sort_values(ascending=False)
    neighbors = set(ranked.head(k).index)
    hit = neighbors & relevant_set
    if hit:
        hits.append(name)
        continue
    # near miss: is a relevant problem ranked just outside top k (11-20)?
    top20 = set(ranked.head(20).index)
    if top20 & relevant_set:
        near_misses.append(name)
    else:
        deep_misses.append(name)

print(f"hits={len(hits)} near_misses={len(near_misses)} deep_misses={len(deep_misses)} total={len(gt_filtered)}")
print()
print("=== DEEP MISSES (relevant problem not even in top 20) ===")
for name in deep_misses:
    rel = gt_filtered[name]
    ranked = blend[name].drop(index=name).sort_values(ascending=False)
    print(f"\n{name}  (relevant: {rel})")
    for r in rel:
        rank = list(ranked.index).index(r) + 1 if r in ranked.index else "N/A"
        score = ranked.get(r, float("nan"))
        print(f"    -> {r}: rank={rank} score={score:.3f}")
