"""
Regenerates abstracts using ONLY openai/gpt-oss-120b, based on direct
evidence that the smaller 20b model (first in the default MODEL_CHAIN) can
silently produce plausible-but-incomplete abstracts for genuinely
multi-phase code: on best-time-to-buy-and-sell-stock-iii, 20b's abstract
(at both the original 2-4 sentence cap AND a relaxed unlimited-length
version) completely dropped the second of two transaction phases; 120b, at
the exact same low reasoning effort and the exact same unmodified prompt,
described both phases correctly on the first try. So this isn't a prompt
issue -- it's specifically that 20b isn't reliably capable of tracing
through less-obvious multi-part anonymized code.

Forces every generation through 120b specifically, with NO fallback to
other models -- if 120b is rate-limited, the run stops rather than falling
back to a weaker model (a fallback would silently reintroduce the exact
class of bug this run exists to fix/measure).

Writes to separate cache files (hq_abstract_cache_120b.pkl /
hq_mechanism_embedding_cache_120b.pkl) -- does not touch hq_abstract_cache.pkl
or any production cache, so the original 20b-chain-generated abstracts stay
available for a direct before/after comparison on the overlapping subset.

Processes hq_source_code_cache names in the same alphabetical order as the
original run_hq_abstract_generation.py, so whatever gets through before
hitting quota has maximum overlap with the already-generated 667 -- that
overlap is exactly the comparison set we care about.

Usage:
    python run_hq_abstract_generation_120b.py
"""

import pickle

from sentence_transformers import SentenceTransformer

import structural_features as sf
import scratch_code_only_abstracts as coa
from scratch_code_embedding_test import anonymize_code

# Force single-model, no fallback -- see module docstring for why.
coa.MODEL_CHAIN = ["openai/gpt-oss-120b"]

HQ_SOURCE_PATH = "hq_source_code_cache.pkl"
CACHE_PATH = "hq_abstract_cache_120b.pkl"
EMB_CACHE_PATH = "hq_mechanism_embedding_cache_120b.pkl"
CHECKPOINT_EVERY = 20


def load(path, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    hq_source = load(HQ_SOURCE_PATH, {})
    print(f"Higher-quality source pool: {len(hq_source)} problems")

    cache = load(CACHE_PATH, {})
    mech_emb = load(EMB_CACHE_PATH, {})
    print(f"Already generated (120b): {len(cache)}")

    remaining = [n for n in sorted(hq_source) if n not in cache]
    print(f"Remaining to attempt: {len(remaining)}")

    exhausted_models = set()
    n_done, n_failed, n_unusable = 0, 0, 0
    stopped_early = False
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    for i, name in enumerate(remaining):
        if exhausted_models:
            stopped_early = True
            break

        entry = hq_source[name]
        code = entry["code"]
        if not sf.is_usable_solution_code(code):
            n_unusable += 1
            continue
        code = anonymize_code(code)

        abstract = coa.get_code_only_abstract(code, exhausted_models)
        if abstract:
            cache[name] = abstract
            mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
            n_done += 1
        else:
            n_failed += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save(cache, CACHE_PATH)
            save(mech_emb, EMB_CACHE_PATH)
            print(f"  ...{i + 1}/{len(remaining)} processed (done={n_done}, failed={n_failed}, "
                  f"unusable={n_unusable}, total cached={len(cache)}, exhausted={sorted(exhausted_models)})")

    save(cache, CACHE_PATH)
    save(mech_emb, EMB_CACHE_PATH)
    pct = 100 * len(cache) / len(hq_source)
    print(f"\nDone{' (stopped early -- 120b exhausted, no fallback by design)' if stopped_early else ''}. "
          f"{n_done} new, {len(cache)} total cached ({pct:.1f}% of hq source pool), "
          f"{n_failed} failures, {n_unusable} unusable source skipped.")


if __name__ == "__main__":
    main()
