"""
Samples current MISSES on the best checkpoint (combined+mechanism,
alpha=0.75, legitimacy-filtered ground truth) and prints everything needed
to manually attribute each miss: the query's mechanism abstract, its
ground-truth partner(s)' abstracts, and what we recommended INSTEAD (top-5
with their abstracts) -- so each miss can be classified as instance-diversity
(partner uses a genuinely different valid technique), a real system miss
(abstracts actually look similar; ranking/embedding failed), an abstract-
quality problem (one side's abstract is inaccurate), or a questionable
ground-truth edge (despite passing the legitimacy filter).

Usage:
    python scratch_miss_diagnosis.py [n_samples]
"""

import pickle
import random
import sys

import numpy as np
import pandas as pd

import categorical_similarity as cs
import evaluate_categorical_similarity as ev

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 18


def mech_text(abstracts, name):
    a = abstracts.get(name)
    if not a:
        return "(no abstract)"
    techniques = ", ".join(t["name"] for t in a.get("techniques", []))
    return f"[{techniques}]\n    {a.get('mechanism', '')}"


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

    base = 0.75 * mechanism_sim + 0.25 * combined

    ground_truth_full = ev.load_similar_questions_ground_truth(df)
    ground_truth = {name: [n for n in rel if n in pool] for name, rel in ground_truth_full.items() if name in pool}
    ground_truth = {name: rel for name, rel in ground_truth.items() if rel}
    legitimacy = ev.load_ground_truth_legitimacy()
    gtf = ev.filter_ground_truth_by_legitimacy(ground_truth, legitimacy)

    misses = []
    for name, relevant in gtf.items():
        if name not in base.index:
            continue
        neighbors = list(base[name].drop(index=name).sort_values(ascending=False).head(10).index)
        hits = set(neighbors) & set(relevant)
        if not hits:
            misses.append((name, relevant, neighbors))

    print(f"Total misses (hit_rate@10 failures) on filtered ground truth: {len(misses)} / {len(gtf)}")
    random.seed(1)
    sample = random.sample(misses, min(N_SAMPLES, len(misses)))

    for name, relevant, neighbors in sample:
        print("\n" + "=" * 100)
        print(f"QUERY: {name}")
        print(f"  {mech_text(abstracts, name)}")
        print(f"\n  GROUND-TRUTH PARTNER(S) (legitimacy-filtered, should have ranked in top 10):")
        for r in relevant:
            label = legitimacy.get(frozenset((name, r)), "?")
            rank = neighbors.index(r) + 1 if r in neighbors else None
            in_pool = r in pool
            print(f"    - {r}  [{label}]  (rank given: {rank if rank else 'not in top-10' if in_pool else 'NOT IN POOL'})")
            if in_pool:
                print(f"      {mech_text(abstracts, r)}")
        print(f"\n  TOP-5 RECOMMENDED INSTEAD:")
        for n in neighbors[:5]:
            score = base.loc[name, n]
            print(f"    - {n}  (score={score:.3f})")
            print(f"      {mech_text(abstracts, n)}")


if __name__ == "__main__":
    main()
