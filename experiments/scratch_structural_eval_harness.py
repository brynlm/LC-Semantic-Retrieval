"""
Coverage + precision evaluation harness for the structural (CFG/DFG-derived)
similarity signal, built BEFORE the phase-1 CFG/DFG schema additions
(REPAIRS/GROWS, RECURRENCE_COMBINATOR, RECURSION_STYLE) land, per the
project's established discipline of measuring before trusting an aggregate
score (see the retracted precision@K calibration, and the tag-purity
spot-checks) rather than building several detectors and evaluating at the
end. Runs unchanged against whatever structural_features.to_tag_set produces,
so re-running this after phase-1 lands gives a direct before/after read.

Reuses existing, already-validated infrastructure rather than inventing new
machinery:
  - structural_corroboration.richness()/RICHNESS_THRESHOLD for the coverage
    gate (empirically validated: below richness=8, structural top-K pairs
    score BELOW the random-pair baseline on mechanism-embedding agreement).
  - categorical_similarity.tag_similarity_matrix (IDF-weighted Dice) for
    structural-tag similarity, same machinery as every other tag-based
    signal in this project.
  - mechanism_embedding_cache.pkl (production abstracts) as the established
    trusted signal for the disagreement/gap-hunting comparison, per
    code_only_vs_production_verdict.md's verdict that production wins on
    ranking.

Produces three things:
  1. Coverage report (printed): what fraction of the pool clears the
     richness gate, at several threshold values for context.
  2. /tmp/structural_audit_direct.txt -- a stratified sample (by structural
     score band) of each rich query's own top-1 pick among other rich
     candidates, with code for both sides, for DIRECT hand-judging of
     whether high structural similarity tracks genuine mechanism match.
     (Not a calibration curve -- read by a human, same as every other
     precision number trusted in this project.)
  3. /tmp/structural_audit_disagreements.txt -- rich queries where the
     structural top-1 pick (among rich candidates) differs from the
     mechanism-embedding top-1 pick over the SAME rich candidate pool. This
     is the gap-hunting set: a disagreement is exactly where a missing
     schema element would most likely explain the difference.

Usage:
    python scratch_structural_eval_harness.py
"""

import pickle

import numpy as np
import pandas as pd

import categorical_similarity as cs
import structural_corroboration as sc
import structural_features as sf

STRUCTURAL_TAGS_PATH = "/tmp/structural_tags.pkl"
STRUCTURAL_SIM_PATH = "/tmp/structural_similarity_matrix.pkl"
MECHANISM_EMBEDDING_CACHE_PATH = "mechanism_embedding_cache.pkl"

DIRECT_AUDIT_PATH = "/tmp/structural_audit_direct.txt"
DISAGREEMENT_AUDIT_PATH = "/tmp/structural_audit_disagreements.txt"

N_DIRECT_SAMPLE = 40
N_DISAGREEMENT_SAMPLE = 40
BANDS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]


def coverage_report(names, structural_tags):
    print("=== Coverage report ===")
    for threshold in [4, 6, 8, 10, 12]:
        n_rich = sum(1 for n in names if sc.richness(n, structural_tags) >= threshold)
        print(f"  richness >= {threshold:2d}: {n_rich:4d}/{len(names)} ({100*n_rich/len(names):.1f}%)")
    sizes = [len(structural_tags.get(n, set())) for n in names]
    print(f"  tag-set size: mean={np.mean(sizes):.1f} median={np.median(sizes):.0f} "
          f"min={min(sizes)} max={max(sizes)}")
    print()


def top1_among(sim: pd.DataFrame, query: str, candidates: list[str]) -> tuple[str, float] | None:
    row = sim.loc[query, candidates].drop(labels=[query], errors="ignore")
    if row.empty:
        return None
    best = row.idxmax()
    return best, float(row[best])


def write_direct_audit(rich_names, structural_sim, name_to_code, path):
    picks = []
    for q in rich_names:
        result = top1_among(structural_sim, q, rich_names)
        if result is None:
            continue
        neighbor, score = result
        picks.append((q, neighbor, score))

    # stratify by score band so bigger bands aren't underrepresented, same
    # discipline as run_filtered_eval.py's precision-by-band fix earlier
    rng = np.random.default_rng(0)
    sampled = []
    per_band = max(1, N_DIRECT_SAMPLE // len(BANDS))
    for lo, hi in BANDS:
        band_picks = [p for p in picks if lo <= p[2] < hi]
        if not band_picks:
            continue
        idx = rng.choice(len(band_picks), size=min(per_band, len(band_picks)), replace=False)
        sampled.extend(band_picks[i] for i in idx)

    with open(path, "w") as f:
        f.write(f"Direct structural-audit sample: {len(sampled)} queries "
                 f"(each query's own top-1 pick among richness>={sc.RICHNESS_THRESHOLD} candidates)\n")
        f.write("Judge each pair MATCH / PARTIAL / MISMATCH independently, same as prior audits.\n")
        f.write("=" * 80 + "\n\n")
        for q, neighbor, score in sorted(sampled, key=lambda p: -p[2]):
            f.write(f"QUERY: {q}   ->   NEIGHBOR: {neighbor}   (structural score={score:.3f})\n")
            f.write("-" * 60 + "\n")
            f.write(f"[{q}]\n{name_to_code.get(q, '<no code>')}\n\n")
            f.write(f"[{neighbor}]\n{name_to_code.get(neighbor, '<no code>')}\n")
            f.write("=" * 80 + "\n\n")
    print(f"Wrote {len(sampled)}-query direct audit sample to {path}")


def write_disagreement_audit(rich_names, structural_sim, mechanism_sim, name_to_code, path):
    disagreements = []
    for q in rich_names:
        s_result = top1_among(structural_sim, q, rich_names)
        m_result = top1_among(mechanism_sim, q, rich_names)
        if s_result is None or m_result is None:
            continue
        s_neighbor, s_score = s_result
        m_neighbor, m_score = m_result
        if s_neighbor != m_neighbor:
            disagreements.append((q, s_neighbor, s_score, m_neighbor, m_score))

    print(f"Disagreements (structural top-1 != mechanism top-1, among richness>={sc.RICHNESS_THRESHOLD} "
          f"candidates): {len(disagreements)}/{len(rich_names)} rich queries")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(disagreements), size=min(N_DISAGREEMENT_SAMPLE, len(disagreements)), replace=False)
    sampled = [disagreements[i] for i in idx]

    with open(path, "w") as f:
        f.write(f"Structural vs mechanism-embedding disagreement sample: {len(sampled)} queries\n")
        f.write("For each, structural and mechanism-embedding signals pick DIFFERENT top-1 neighbors\n")
        f.write("among the same richness-gated candidate pool. Read both picks and judge which (if\n")
        f.write("either) is the better match -- a case where structural wins is exactly where a\n")
        f.write("missing CFG/DFG schema element would most likely explain the gap.\n")
        f.write("=" * 80 + "\n\n")
        for q, s_neighbor, s_score, m_neighbor, m_score in sampled:
            f.write(f"QUERY: {q}\n")
            f.write(f"  STRUCTURAL pick: {s_neighbor} (score={s_score:.3f})\n")
            f.write(f"  MECHANISM   pick: {m_neighbor} (score={m_score:.3f})\n")
            f.write("-" * 60 + "\n")
            f.write(f"[{q}]\n{name_to_code.get(q, '<no code>')}\n\n")
            f.write(f"[{s_neighbor}] (structural pick)\n{name_to_code.get(s_neighbor, '<no code>')}\n\n")
            f.write(f"[{m_neighbor}] (mechanism pick)\n{name_to_code.get(m_neighbor, '<no code>')}\n")
            f.write("=" * 80 + "\n\n")
    print(f"Wrote {len(sampled)}-query disagreement audit sample to {path}")


def main():
    df = cs.load_dataset()
    # MUST match scratch_structural_similarity.py's code preference exactly
    # (official editorial code over the dataset's own `code` column) -- the
    # similarity scores are computed from whichever code that script fed to
    # extract_features, and 311/639 pool problems have official-editorial
    # code that DIFFERS from the dataset's own column. Displaying the wrong
    # one here means auditing a solution that isn't what was actually
    # scored -- a real bug, not a cosmetic one, found while re-checking a
    # confusing audit result (letter-tile-possibilities looked like a
    # different, unrelated solution than what its tags implied).
    with open("official_editorial_code_cache.pkl", "rb") as f:
        official_code = pickle.load(f)
    name_to_code = {}
    for name, code in zip(df["name"], df["code"]):
        editorial = official_code.get(name)
        name_to_code[name] = editorial if editorial is not None and sf.is_usable_solution_code(editorial) else code

    with open(STRUCTURAL_TAGS_PATH, "rb") as f:
        structural_tags = pickle.load(f)
    structural_sim = pd.read_pickle(STRUCTURAL_SIM_PATH)
    with open(MECHANISM_EMBEDDING_CACHE_PATH, "rb") as f:
        mechanism_embeddings = pickle.load(f)

    names = list(structural_sim.index)
    coverage_report(names, structural_tags)

    rich_names = [n for n in names if sc.richness(n, structural_tags) >= sc.RICHNESS_THRESHOLD]
    print(f"Using {len(rich_names)} richness-gated problems for audit sampling.\n")

    vectors = np.stack([mechanism_embeddings[n] for n in rich_names])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    mechanism_sim = pd.DataFrame(vectors @ vectors.T, index=rich_names, columns=rich_names)
    structural_sim_rich = structural_sim.loc[rich_names, rich_names]

    write_direct_audit(rich_names, structural_sim_rich, name_to_code, DIRECT_AUDIT_PATH)
    write_disagreement_audit(rich_names, structural_sim_rich, mechanism_sim, name_to_code, DISAGREEMENT_AUDIT_PATH)


if __name__ == "__main__":
    main()
