import pickle
import random
import numpy as np
import pandas as pd
import categorical_similarity as cs
import evaluate_categorical_similarity as ev

df = cs.load_dataset()
with open("abstract_discovery_cache.pkl", "rb") as f:
    abstracts = pickle.load(f)
with open("mechanism_embedding_cache.pkl", "rb") as f:
    mech_emb = pickle.load(f)
with open("/tmp/structural_tags.pkl", "rb") as f:
    structural_tags = pickle.load(f)
structural_sim = pd.read_pickle("/tmp/structural_similarity_matrix.pkl")
legitimacy = ev.load_ground_truth_legitimacy()

pool = sorted(set(abstracts) & set(mech_emb) & set(df["name"]))
names = pool
vectors = np.stack([mech_emb[n] for n in names])
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
cos_sim = vectors @ vectors.T
mechanism_sim = pd.DataFrame(cos_sim, index=names, columns=names)

# top structural-similarity pairs (excluding self, dedup via i<j)
pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        s = structural_sim.loc[a, b]
        if s > 0:
            pairs.append((s, a, b))
pairs.sort(reverse=True)
print(f"Total pairs with nonzero structural similarity: {len(pairs)} / {len(names)*(len(names)-1)//2}")

top_n = 200
top_pairs = pairs[:top_n]
top_mech_scores = [mechanism_sim.loc[a, b] for _, a, b in top_pairs]
print(f"\nTop {top_n} structural-similarity pairs: mean mechanism-sim={np.mean(top_mech_scores):.3f}, median={np.median(top_mech_scores):.3f}")

random.seed(0)
baseline_scores = []
for _ in range(2000):
    a, b = random.sample(names, 2)
    baseline_scores.append(mechanism_sim.loc[a, b])
print(f"Random baseline pairs: mean mechanism-sim={np.mean(baseline_scores):.3f}, median={np.median(baseline_scores):.3f}")

# also check: does this hold even restricting to pairs where both sides have >= 5 non-trivial tags?
def n_tags(name):
    return len(structural_tags.get(name, set()))

filtered_pairs = [(s, a, b) for s, a, b in pairs if n_tags(a) >= 8 and n_tags(b) >= 8]
filtered_top = filtered_pairs[:top_n]
filtered_scores = [mechanism_sim.loc[a, b] for _, a, b in filtered_top]
print(f"\nTop {top_n} structural pairs RESTRICTED to tag-rich problems (>=8 tags each): mean mechanism-sim={np.mean(filtered_scores):.3f}")
print(f"(n eligible pairs after restriction: {len(filtered_pairs)})")

print("\n=== Top 30 structural-similarity pairs, with mechanism-sim and GT label ===")
for s, a, b in top_pairs[:30]:
    m = mechanism_sim.loc[a, b]
    label = legitimacy.get(frozenset((a, b)), "no-GT-edge")
    print(f"struct={s:.3f}  mech={m:.3f}  {a} <-> {b}  [{label}]")

# correlation with legitimacy labels specifically: for LEGITIMATE/LOOSE pairs
# reachable in-pool, what's their structural similarity vs SPURIOUS ones?
legit_scores, spurious_scores = [], []
for pair, label in legitimacy.items():
    a, b = tuple(pair)
    if a not in pool or b not in pool:
        continue
    s = structural_sim.loc[a, b]
    if label in ("LEGITIMATE", "LOOSE"):
        legit_scores.append(s)
    elif label in ("SPURIOUS", "SPURIOUS_NAME_FAMILY"):
        spurious_scores.append(s)
print(f"\nStructural similarity: LEGITIMATE/LOOSE pairs mean={np.mean(legit_scores):.3f} (n={len(legit_scores)}), SPURIOUS pairs mean={np.mean(spurious_scores):.3f} (n={len(spurious_scores)})")
