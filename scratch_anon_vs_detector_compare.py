"""
Direct A/B comparison: for a fixed set of already-generated, detector-filtered
(raw-code input + banned-word-retry) abstracts, regenerate the SAME problems
using the anonymized-code approach into a SEPARATE cache
(anon_compare_abstract_cache.pkl / anon_compare_mechanism_embedding_cache.pkl
-- never touches code_only_abstract_cache.pkl, which the background full-run
is actively writing to), then compare pairwise similarity for known family
cases under both approaches side by side.

This answers the actual open question: the two approaches are meant to solve
the same problem, but "meant to" isn't "measured to" -- if there's no
material difference, the reactive detector-filtered abstracts already in hand
are good enough and a full regeneration isn't worth the tokens; if there is a
measurable difference, that's the case for regenerating everything.

Usage:
    python scratch_anon_vs_detector_compare.py
"""

import pickle

from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import scratch_code_only_abstracts as sc
from scratch_code_embedding_test import anonymize_code

COMPARE_ABSTRACT_PATH = "anon_compare_abstract_cache.pkl"
COMPARE_EMB_PATH = "anon_compare_mechanism_embedding_cache.pkl"

FAMILIES = {
    "course-schedule": ["course-schedule", "course-schedule-ii", "course-schedule-iii"],
    "basic-calculator": ["basic-calculator", "basic-calculator-ii", "basic-calculator-iii", "basic-calculator-iv"],
    "majority-element": ["majority-element", "majority-element-ii"],
    "decode-ways": ["decode-ways", "decode-ways-ii"],
    "erect-the-fence": ["erect-the-fence", "erect-the-fence-ii"],
}


def main():
    df = cs.load_dataset()
    name_to_code = dict(zip(df["name"], df["code"]))
    with open("official_editorial_code_cache.pkl", "rb") as f:
        official_code = pickle.load(f)
    with open("code_only_abstract_cache.pkl", "rb") as f:
        detector_cache = pickle.load(f)
    with open("code_only_mechanism_embedding_cache.pkl", "rb") as f:
        detector_emb = pickle.load(f)

    targets = sorted({n for names in FAMILIES.values() for n in names})
    targets = [n for n in targets if n in detector_cache]
    print(f"Comparing {len(targets)} names present in the detector-filtered cache: {targets}")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    anon_cache, anon_emb = {}, {}
    exhausted = set()
    for name in targets:
        code = official_code.get(name, name_to_code[name])
        anon_code = anonymize_code(code)
        a = sc.get_code_only_abstract(anon_code, exhausted)
        if a:
            anon_cache[name] = a
            anon_emb[name] = embed_model.encode(a["mechanism"], normalize_embeddings=True)
            print(f"  [{name}] anonymized-abstract generated")
        else:
            print(f"  [{name}] FAILED to generate anonymized abstract")

    with open(COMPARE_ABSTRACT_PATH, "wb") as f:
        pickle.dump(anon_cache, f)
    with open(COMPARE_EMB_PATH, "wb") as f:
        pickle.dump(anon_emb, f)

    def sim(cache, a, b):
        va = cache[a] / (cache[a] ** 2).sum() ** 0.5
        vb = cache[b] / (cache[b] ** 2).sum() ** 0.5
        return float(va @ vb)

    print("\n=== Pairwise similarity: detector-filtered (raw-code input) vs anonymized-code input ===")
    for fam, names in FAMILIES.items():
        names = [n for n in names if n in detector_cache and n in anon_cache]
        if len(names) < 2:
            continue
        print(f"\n{fam}:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                det_s = sim(detector_emb, a, b)
                anon_s = sim(anon_emb, a, b)
                print(f"  {a} <-> {b}:  detector-filtered={det_s:.3f}  anonymized={anon_s:.3f}  (delta={anon_s - det_s:+.3f})")


if __name__ == "__main__":
    main()
