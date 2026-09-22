import pickle
import pandas as pd
import categorical_similarity as cs

df = cs.load_dataset()
with open("abstract_discovery_cache.pkl", "rb") as f:
    abstracts = pickle.load(f)
tags_map = dict(zip(df["name"], df["tags"]))

final_sim = pd.read_pickle("/tmp/final_similarity_matrix.pkl")

QUERIES = [
    "two-sum", "climbing-stairs", "course-schedule-ii", "merge-intervals",
    "longest-increasing-subsequence", "number-of-provinces", "reverse-pairs",
    "word-break", "minimum-height-trees", "binary-tree-inorder-traversal",
]

def fmt(name):
    ab = abstracts.get(name, {})
    techniques = "; ".join(t["name"] for t in ab.get("techniques", []))
    return f"{name} | tags={tags_map.get(name)} | techniques=[{techniques}]"

with open("/tmp/recommendation_review.txt", "w") as f:
    for q in QUERIES:
        if q not in final_sim.index:
            f.write(f"QUERY {q}: NOT IN POOL\n\n")
            continue
        f.write("=" * 100 + "\n")
        f.write(f"QUERY: {fmt(q)}\n")
        f.write(f"  mechanism: {abstracts[q]['mechanism']}\n")
        f.write("-" * 100 + "\n")
        top10 = final_sim[q].drop(index=q).sort_values(ascending=False).head(10)
        for rank, (name, score) in enumerate(top10.items(), 1):
            f.write(f"  {rank}. [{score:.3f}] {fmt(name)}\n")
            f.write(f"       mechanism: {abstracts[name]['mechanism']}\n")
        f.write("\n")

print("done")
