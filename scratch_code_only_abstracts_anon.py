"""
Same code-only abstract generation as scratch_code_only_abstracts.py, but
feeds the LLM identifier-anonymized code (function name -> "solve", every
local variable/parameter -> v0, v1, v2... by first appearance) instead of
the original code. Spot-tested on stone-game-ii/iii/vii and
parallel-courses-ii: all 4 succeeded clean on the FIRST attempt, versus the
raw-code approach which needed many retries and often failed outright on
these exact problems (their code literally contains identifiers like
`stoneGameII`, `piles`, `stones`). Structural prevention (the words aren't in
the input at all) beats reactive detection (banned-word-list-and-retry on
the output) -- cheaper on rate-limit budget too, since it wastes far fewer
attempts.

Resumes from and writes to the SAME caches as scratch_code_only_abstracts.py
(code_only_abstract_cache.pkl / code_only_mechanism_embedding_cache.pkl) --
this is meant to finish that same batch, not start a separate one, so
whatever's already cached from the raw-code run is kept as-is and only the
remaining/failed names are attempted here.

Usage:
    python scratch_code_only_abstracts_anon.py
"""

import os
import pickle

from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import scratch_code_only_abstracts as sc
from scratch_code_embedding_test import anonymize_code

CHECKPOINT_EVERY = 25


def main():
    df = cs.load_dataset()
    name_to_code = dict(zip(df["name"], df["code"]))
    with open("official_editorial_code_cache.pkl", "rb") as f:
        official_code = pickle.load(f)

    import csv
    with open("/tmp/audit_verdicts.csv") as fcsv:
        audit_rows = list(csv.DictReader(fcsv))
    seen = set()
    names_needed = set()
    for r in audit_rows:
        names_needed.add(r["query"])
        names_needed.add(r["neighbor"])
    names_needed &= set(name_to_code)
    names_list = sorted(names_needed)

    cache = {}
    if os.path.exists(sc.CACHE_PATH):
        with open(sc.CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
    print(f"Resuming from {len(cache)} already-cached abstracts")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    mech_emb = {}
    if os.path.exists(sc.EMB_CACHE_PATH):
        with open(sc.EMB_CACHE_PATH, "rb") as f:
            mech_emb = pickle.load(f)

    exhausted_models = set()
    n_done, n_failed = 0, 0
    remaining = [n for n in names_list if n not in cache]
    print(f"Remaining to attempt (anonymized-code approach): {len(remaining)}")

    for i, name in enumerate(remaining):
        code = official_code.get(name, name_to_code[name])
        anon = anonymize_code(code)
        abstract = sc.get_code_only_abstract(anon, exhausted_models)
        if abstract:
            cache[name] = abstract
            mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
            n_done += 1
        else:
            n_failed += 1
            print(f"  [{name}] failed / still leaked after retries (anonymized code)")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(sc.CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
            with open(sc.EMB_CACHE_PATH, "wb") as f:
                pickle.dump(mech_emb, f)
            print(f"  ...{i + 1}/{len(remaining)} processed (done={n_done}, failed={n_failed})")

    with open(sc.CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    with open(sc.EMB_CACHE_PATH, "wb") as f:
        pickle.dump(mech_emb, f)
    print(f"\nDone. {n_done} new, {len(cache)} total cached, {n_failed} failures.")


if __name__ == "__main__":
    main()
