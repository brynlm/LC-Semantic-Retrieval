"""
Stratified sample of (query, top-1 match) pairs across top-1-score bands, for
a manual, ground-truth-independent quality audit: is the model's single best
recommendation for this query actually a genuine match? Ground truth is
cross-referenced only as a secondary suspicion flag, per the brief -- primary
judgment is direct abstract-vs-abstract reading.

Dumps a compact per-item block (technique tags + truncated mechanism for both
query and its top-1 neighbor, plus GT legitimacy label if any) to a text file
for manual review, not the terminal, since the full sample is large.

Usage:
    python scratch_quality_audit_sample.py
"""

import pickle
import random

import numpy as np

import categorical_similarity as cs
import evaluate_categorical_similarity as ev

BANDS = [(0.75, 1.01, 35), (0.65, 0.75, 40), (0.55, 0.65, 45), (0.45, 0.55, 35), (0.35, 0.45, 14)]
MECH_TRUNCATE = 200


def mech_text(abstracts, name):
    a = abstracts.get(name)
    if not a:
        return "(no abstract)"
    techniques = ", ".join(t["name"] for t in a.get("techniques", []))
    mech = a.get("mechanism", "")
    if len(mech) > MECH_TRUNCATE:
        mech = mech[:MECH_TRUNCATE] + "..."
    return f"[{techniques}] {mech}"


def main():
    df = cs.load_dataset()
    with open("abstract_discovery_cache.pkl", "rb") as f:
        abstracts = pickle.load(f)
    base = pickle.load(open("/tmp/full_pool_similarity_matrix.pkl", "rb"))
    top1_scores = pickle.load(open("/tmp/top1_scores.pkl", "rb"))
    names = list(base.index)

    legitimacy = ev.load_ground_truth_legitimacy()
    gt = ev.load_similar_questions_ground_truth(df)

    top1_name = {}
    for n in names:
        row = base[n].drop(index=n)
        top1_name[n] = row.idxmax()

    random.seed(2)
    sample_items = []
    for lo, hi, k in BANDS:
        band_names = [n for n in names if lo <= top1_scores[n] < hi]
        random.shuffle(band_names)
        chosen = band_names[:k]
        for n in chosen:
            sample_items.append((n, top1_scores[n], top1_name[n]))

    print(f"Total sampled: {len(sample_items)}")

    with open("/tmp/quality_audit_dump.txt", "w") as f:
        for query, score, neighbor in sample_items:
            label = legitimacy.get(frozenset((query, neighbor)))
            is_declared_gt = neighbor in gt.get(query, []) or query in gt.get(neighbor, [])
            gt_tag = label or ("RAW_GT_UNJUDGED" if is_declared_gt else "no-GT-edge")
            f.write(f"SCORE={score:.3f}  GT=[{gt_tag}]\n")
            f.write(f"  Q: {query}\n")
            f.write(f"     {mech_text(abstracts, query)}\n")
            f.write(f"  -> {neighbor}\n")
            f.write(f"     {mech_text(abstracts, neighbor)}\n")
            f.write("\n")

    print("Dumped to /tmp/quality_audit_dump.txt")


if __name__ == "__main__":
    main()
