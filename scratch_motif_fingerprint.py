import pickle
import categorical_similarity as cs
import motif_detectors as md

df = cs.load_dataset()
with open("abstract_discovery_cache.pkl", "rb") as f:
    abstracts = pickle.load(f)
with open("mechanism_embedding_cache.pkl", "rb") as f:
    mech_emb = pickle.load(f)
with open("official_editorial_code_cache.pkl", "rb") as f:
    official_code = pickle.load(f)

pool = sorted(set(abstracts) & set(mech_emb) & set(df["name"]))
name_to_code = dict(zip(df["name"], df["code"]))

DETECTORS = {
    "monotonic_stack": md.detect_monotonic_stack,
    "binary_search": md.detect_binary_search,
    "two_pointer_convergence": md.detect_two_pointer_convergence,
}

fingerprints = {}
code_source = {}
for name in pool:
    code = official_code.get(name, name_to_code.get(name))
    code_source[name] = "official" if name in official_code else "dataset"
    motifs = set()
    for motif_name, detector in DETECTORS.items():
        try:
            if detector(code):
                motifs.add(motif_name)
        except SyntaxError:
            pass
    fingerprints[name] = motifs

with open("/tmp/motif_fingerprints.pkl", "wb") as f:
    pickle.dump(fingerprints, f)

any_motif = {n: m for n, m in fingerprints.items() if m}
print(f"Pool size: {len(pool)}")
print(f"Problems with >=1 detected motif: {len(any_motif)}")
from collections import Counter
motif_counts = Counter()
for m in fingerprints.values():
    for x in m:
        motif_counts[x] += 1
print(motif_counts)
print()
for n, m in sorted(any_motif.items()):
    print(f"{n}: {m} ({code_source[n]})")
