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
with open("/tmp/motif_fingerprints.pkl", "rb") as f:
    fingerprints = pickle.load(f)
legitimacy = ev.load_ground_truth_legitimacy()

pool = sorted(set(abstracts) & set(mech_emb) & set(df["name"]))
names = pool
vectors = np.stack([mech_emb[n] for n in names])
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
cos_sim = vectors @ vectors.T
mechanism_sim = pd.DataFrame(cos_sim, index=names, columns=names)

# every same-motif pair (excluding self)
motif_pairs = []
by_motif = {}
for n, motifs in fingerprints.items():
    for m in motifs:
        by_motif.setdefault(m, []).append(n)

for motif, members in by_motif.items():
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            motif_pairs.append((motif, a, b, mechanism_sim.loc[a, b]))

print(f"Total same-motif pairs: {len(motif_pairs)}")
motif_scores = [s for _, _, _, s in motif_pairs]
print(f"Motif-sharing pairs: mean sim={np.mean(motif_scores):.3f}, median={np.median(motif_scores):.3f}")

# random baseline: same number of random pairs from the pool
random.seed(0)
baseline_scores = []
for _ in range(2000):
    a, b = random.sample(names, 2)
    baseline_scores.append(mechanism_sim.loc[a, b])
print(f"Random baseline pairs: mean sim={np.mean(baseline_scores):.3f}, median={np.median(baseline_scores):.3f}")

print()
print("=== All same-motif pairs, sorted by similarity ===")
for motif, a, b, s in sorted(motif_pairs, key=lambda x: -x[3]):
    label = legitimacy.get(frozenset((a, b)), "no-GT-edge")
    print(f"[{motif}] {a} <-> {b}: sim={s:.3f}  [{label}]")
