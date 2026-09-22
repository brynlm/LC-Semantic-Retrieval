"""
Full-coverage extension of the manual quality audit: dumps the remaining
query->top-1-match pairs (every pool problem not already covered by the
initial 169-pair stratified sample) in the same compact format for manual
review, split into numbered chunks so each is a manageable single read.

Usage:
    python scratch_full_audit_remaining.py
"""

import pickle

import categorical_similarity as cs
import evaluate_categorical_similarity as ev

MECH_TRUNCATE = 200
CHUNK_SIZE = 120


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

    already_done = set()
    with open("/tmp/audit_verdicts.csv") as f:
        next(f)
        for line in f:
            q = line.split(",")[0]
            already_done.add(q)

    legitimacy = ev.load_ground_truth_legitimacy()
    gt = ev.load_similar_questions_ground_truth(df)

    top1_name = {}
    for n in names:
        row = base[n].drop(index=n)
        top1_name[n] = row.idxmax()

    remaining = sorted((n for n in names if n not in already_done), key=lambda n: -top1_scores[n])
    print(f"Remaining to audit: {len(remaining)} / {len(names)} total pool")

    for chunk_i in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[chunk_i:chunk_i + CHUNK_SIZE]
        path = f"/tmp/full_audit_chunk_{chunk_i // CHUNK_SIZE + 1}.txt"
        with open(path, "w") as f:
            for query in chunk:
                neighbor = top1_name[query]
                score = top1_scores[query]
                label = legitimacy.get(frozenset((query, neighbor)))
                is_declared_gt = neighbor in gt.get(query, []) or query in gt.get(neighbor, [])
                gt_tag = label or ("RAW_GT_UNJUDGED" if is_declared_gt else "no-GT-edge")
                f.write(f"SCORE={score:.3f}  GT=[{gt_tag}]\n")
                f.write(f"  Q: {query}\n")
                f.write(f"     {mech_text(abstracts, query)}\n")
                f.write(f"  -> {neighbor}\n")
                f.write(f"     {mech_text(abstracts, neighbor)}\n")
                f.write("\n")
        print(f"Wrote {path} ({len(chunk)} pairs)")


if __name__ == "__main__":
    main()
