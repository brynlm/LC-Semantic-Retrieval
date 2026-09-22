"""
Grows the classified pool (currently 484/2641 = 18% of the full dataset)
beyond the problems already attempted, rather than continuing to refine
abstracts for problems already in it.

For each newly-selected problem: scrapes the official editorial solution
(leetcode_editorial_scraper), falling back to the dataset's own code if no
editorial is found, generates a mechanism abstract from whichever code was
used, and embeds it. Reuses the seed+neighbor ground-truth sampling from
evaluate_categorical_similarity (~5-6x more evaluable-query-efficient per
classification than tag-stratified sampling) and the bounded retry-with-
backoff model-fallback chain from run_editorial_abstract_finish.py (Groq's
429s here are short tokens-per-minute cooldowns, not daily-quota exhaustion --
blacklisting a model on the first one cascades through the whole chain in
seconds).

Resumable: every cache is keyed by problem name and checkpointed periodically,
reruns skip names already in fetch_status.

Usage:
    python run_pool_expansion.py [target_new]
"""

import os
import pickle
import sys
import time

from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import evaluate_categorical_similarity as ev
import leetcode_editorial_scraper as les
import test_approach_abstract as taa

MODEL_CHAIN = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-safeguard-20b"]

ABSTRACT_CACHE_PATH = "abstract_discovery_cache.pkl"
MECHANISM_EMBEDDING_CACHE_PATH = "mechanism_embedding_cache.pkl"
FETCH_STATUS_CACHE_PATH = "editorial_fetch_status_cache.pkl"
OFFICIAL_CODE_CACHE_PATH = "official_editorial_code_cache.pkl"

CHECKPOINT_EVERY = 20


def load(path, default):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def get_abstract_with_backoff(problem: dict, exhausted_models: set) -> dict | None:
    """Same bounded retry-then-fallback-to-next-model logic as
    run_editorial_abstract_finish.py: a 429 here is almost always a short
    tokens-per-minute cooldown (retry-after of a few seconds), not daily-quota
    exhaustion, so only blacklist a model for the rest of THIS run if the
    suggested wait looks like real exhaustion (>30s)."""
    for model in MODEL_CHAIN:
        if model in exhausted_models:
            continue
        taa.MODEL = model
        for attempt in range(4):
            try:
                return taa.get_abstract(problem)
            except Exception as e:
                headers = getattr(getattr(e, "response", None), "headers", None)
                retry_after = float(headers.get("retry-after", 0)) if headers else 0
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                if is_rate_limit and retry_after and retry_after <= 30 and attempt < 3:
                    time.sleep(retry_after + 1)
                    continue
                if is_rate_limit:
                    exhausted_models.add(model)
                break
    return None


def main():
    target_new = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    df = cs.load_dataset()
    name_to_code = dict(zip(df["name"], df["code"]))
    abstracts = load(ABSTRACT_CACHE_PATH, {})
    mech_emb = load(MECHANISM_EMBEDDING_CACHE_PATH, {})
    fetch_status = load(FETCH_STATUS_CACHE_PATH, {})
    official_code = load(OFFICIAL_CODE_CACHE_PATH, {})

    gt = ev.load_similar_questions_ground_truth(df)
    sample = ev.ground_truth_seed_sample(df, gt, target_new=target_new, already_cached=set(fetch_status), seed=1)
    pending = sorted(n for n in sample if n not in fetch_status)
    print(f"Pool expansion batch: {len(pending)} new problems (pool currently {len(abstracts)}/{len(df)})")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    exhausted_models = set()
    n_found, n_not_found, n_abstract_ok, n_abstract_failed = 0, 0, 0, 0

    for i, name in enumerate(pending):
        code = les.fetch_official_python_solution(name)
        if code:
            official_code[name] = code
            fetch_status[name] = "found"
            n_found += 1
        else:
            fetch_status[name] = "not_found"
            n_not_found += 1
        code_for_abstract = code or name_to_code.get(name)

        row = df[df["name"] == name].iloc[0]
        problem = {"name": name, "description": row["description"], "code": code_for_abstract}
        abstract = get_abstract_with_backoff(problem, exhausted_models)
        if abstract:
            abstracts[name] = abstract
            mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
            n_abstract_ok += 1
        else:
            n_abstract_failed += 1
            fetch_status[name] = fetch_status[name] + "_but_abstract_failed" if code else "not_found"
            print(f"  [{name}] abstract generation failed")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save(official_code, OFFICIAL_CODE_CACHE_PATH)
            save(fetch_status, FETCH_STATUS_CACHE_PATH)
            save(abstracts, ABSTRACT_CACHE_PATH)
            save(mech_emb, MECHANISM_EMBEDDING_CACHE_PATH)
            print(f"  ...{i + 1}/{len(pending)} processed "
                  f"(editorial found={n_found}, not_found={n_not_found}, "
                  f"abstracts ok={n_abstract_ok}, failed={n_abstract_failed})")

    save(official_code, OFFICIAL_CODE_CACHE_PATH)
    save(fetch_status, FETCH_STATUS_CACHE_PATH)
    save(abstracts, ABSTRACT_CACHE_PATH)
    save(mech_emb, MECHANISM_EMBEDDING_CACHE_PATH)
    print(f"\nDone. Pool now {len(abstracts)}/{len(df)}. "
          f"This batch: editorial found={n_found}, not_found={n_not_found}, "
          f"abstracts ok={n_abstract_ok}, failed={n_abstract_failed}.")


if __name__ == "__main__":
    main()
