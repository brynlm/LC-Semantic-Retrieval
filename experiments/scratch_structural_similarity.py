import pickle
import numpy as np
import pandas as pd
import categorical_similarity as cs
import structural_features as sf
import evaluate_categorical_similarity as ev

df = cs.load_dataset()
with open("abstract_discovery_cache.pkl", "rb") as f:
    abstracts = pickle.load(f)
with open("mechanism_embedding_cache.pkl", "rb") as f:
    mech_emb = pickle.load(f)
with open("official_editorial_code_cache.pkl", "rb") as f:
    official_code = pickle.load(f)

pool = sorted(set(abstracts) & set(mech_emb) & set(df["name"]))
name_to_code = dict(zip(df["name"], df["code"]))

structural_tags = {}
errors = 0
fallbacks = 0
for name in pool:
    editorial = official_code.get(name)
    # prefer official-editorial code, but only if it's actually a usable
    # solution -- a handful of cached entries are incomplete prose
    # fragments with no function def at all (see is_usable_solution_code),
    # which would otherwise silently produce meaningless tags
    if editorial is not None and sf.is_usable_solution_code(editorial):
        code = editorial
    else:
        if editorial is not None:
            fallbacks += 1
        code = name_to_code.get(name)
    try:
        feats = sf.extract_features(code)
        structural_tags[name] = sf.to_tag_set(feats)
    except SyntaxError:
        errors += 1
        structural_tags[name] = set()

print(f"Computed structural tags for {len(structural_tags)} problems ({errors} parse errors, "
      f"{fallbacks} fell back to dataset code due to unusable cached editorial code)")
tag_lengths = [len(t) for t in structural_tags.values()]
print(f"Tag-set sizes: mean={np.mean(tag_lengths):.1f}, median={np.median(tag_lengths):.0f}, min={min(tag_lengths)}, max={max(tag_lengths)}")

with open("/tmp/structural_tags.pkl", "wb") as f:
    pickle.dump(structural_tags, f)

structural_df = pd.DataFrame({"name": pool, "tags": [list(structural_tags[n]) for n in pool]})
structural_sim = cs.tag_similarity_matrix(structural_df, idf_weighted=True, metric="dice")
structural_sim.to_pickle("/tmp/structural_similarity_matrix.pkl")
print("Structural similarity matrix built and saved.")
