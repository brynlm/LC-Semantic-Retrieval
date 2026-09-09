"""
Full-pool rollout of official-editorial-code abstract regeneration.

For every problem in the classified pool, tries to fetch its official
LeetCode solution (via leetcode_editorial_scraper, using the user's Premium
session) and, if found, regenerates its abstract from that code (a single
LLM call -- no generate-verify retry loop needed, since official code is
already trustworthy). Problems with no official solution keep their existing
(already prompt-fixed) abstract untouched.

Resumable: fetch status is cached per-name in
editorial_fetch_status_cache.pkl ("found" | "not_found"), so a rerun only
processes names not yet attempted. Progress is saved incrementally.

Usage:
    python run_editorial_rollout.py
"""

import os
import pickle
import time

from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import leetcode_editorial_scraper as les
import test_approach_abstract as taa

taa.MODEL = "openai/gpt-oss-120b"

ABSTRACT_CACHE_PATH = "abstract_discovery_cache.pkl"
MECHANISM_EMBEDDING_CACHE_PATH = "mechanism_embedding_cache.pkl"
FETCH_STATUS_CACHE_PATH = "editorial_fetch_status_cache.pkl"
OFFICIAL_CODE_CACHE_PATH = "official_editorial_code_cache.pkl"

REQUEST_DELAY_S = 0.4  # politeness delay between LeetCode HTTP calls


def load(path, default):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    df = cs.load_dataset()
    abstracts = load(ABSTRACT_CACHE_PATH, {})
    mech_emb = load(MECHANISM_EMBEDDING_CACHE_PATH, {})
    fetch_status = load(FETCH_STATUS_CACHE_PATH, {})
    official_code = load(OFFICIAL_CODE_CACHE_PATH, {})

    pool = sorted(set(abstracts) & set(mech_emb) & set(df["name"]))
    print(f"Pool size: {len(pool)}")

    to_process = [n for n in pool if n not in fetch_status]
    print(f"Already attempted: {len(pool) - len(to_process)}, remaining: {len(to_process)}")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    found_count, not_found_count, regenerated_count = 0, 0, 0
    for i, name in enumerate(to_process):
        try:
            code = les.fetch_official_python_solution(name)
        except Exception as e:
            print(f"  [{name}] fetch error: {str(e)[:150]}")
            code = None
        time.sleep(REQUEST_DELAY_S)

        if code:
            found_count += 1
            official_code[name] = code
            row = df[df["name"] == name].iloc[0]
            problem = {"name": name, "description": row["description"], "code": code}
            try:
                abstract = taa.get_abstract(problem)
                abstracts[name] = abstract
                mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
                regenerated_count += 1
                fetch_status[name] = "found"
            except Exception as e:
                print(f"  [{name}] abstract-gen error (kept old abstract): {str(e)[:150]}")
                fetch_status[name] = "found_but_abstract_failed"
        else:
            not_found_count += 1
            fetch_status[name] = "not_found"

        if (i + 1) % 25 == 0:
            save(abstracts, ABSTRACT_CACHE_PATH)
            save(mech_emb, MECHANISM_EMBEDDING_CACHE_PATH)
            save(fetch_status, FETCH_STATUS_CACHE_PATH)
            save(official_code, OFFICIAL_CODE_CACHE_PATH)
            print(f"  ...{i + 1}/{len(to_process)} processed (found={found_count}, not_found={not_found_count}, regenerated={regenerated_count})")

    save(abstracts, ABSTRACT_CACHE_PATH)
    save(mech_emb, MECHANISM_EMBEDDING_CACHE_PATH)
    save(fetch_status, FETCH_STATUS_CACHE_PATH)
    save(official_code, OFFICIAL_CODE_CACHE_PATH)

    print(f"\nDone. {found_count} official solutions found, {not_found_count} not found, {regenerated_count} abstracts regenerated.")
    print(f"Total status cache: {len(fetch_status)} problems attempted overall.")


if __name__ == "__main__":
    main()
