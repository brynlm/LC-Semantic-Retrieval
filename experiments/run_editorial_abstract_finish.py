"""
Follow-up pass: for problems where fetch_official_python_solution already
succeeded (official_editorial_code_cache.pkl) but abstract generation failed
(fetch_status == "found_but_abstract_failed", usually a rate-limit hit),
retry just the abstract-generation step -- no need to re-scrape LeetCode.

Usage:
    python run_editorial_abstract_finish.py
"""

import os
import pickle
import time

from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import test_approach_abstract as taa

# Tried in order per problem; groq/compound(-mini) are excluded since they
# route through openai/gpt-oss-120b internally (shared, already-exhausted
# quota, not a separate pool), and allam-2-7b is excluded after producing a
# wrong technique ("binary search") for two-sum in a spot check -- not
# trustworthy for this task even though it has quota.
MODEL_CHAIN = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-safeguard-20b"]

ABSTRACT_CACHE_PATH = "abstract_discovery_cache.pkl"
MECHANISM_EMBEDDING_CACHE_PATH = "mechanism_embedding_cache.pkl"
FETCH_STATUS_CACHE_PATH = "editorial_fetch_status_cache.pkl"
OFFICIAL_CODE_CACHE_PATH = "official_editorial_code_cache.pkl"


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

    pending = [n for n, s in fetch_status.items() if s == "found_but_abstract_failed"]
    print(f"Pending abstract regeneration: {len(pending)}")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    done, failed = 0, 0
    exhausted_models = set()
    for i, name in enumerate(pending):
        code = official_code.get(name)
        if not code:
            print(f"  [{name}] no cached code, skipping")
            continue
        row = df[df["name"] == name].iloc[0]
        problem = {"name": name, "description": row["description"], "code": code}

        abstract, last_err = None, None
        for model in MODEL_CHAIN:
            if model in exhausted_models:
                continue
            taa.MODEL = model
            # Groq's 429s here are almost always short tokens-per-minute
            # limits (retry-after of a few seconds), not daily-quota
            # exhaustion -- blacklisting a model on the first one (as this
            # used to do) cascades through the whole MODEL_CHAIN within
            # seconds and stalls the rest of the batch. Retry the same model
            # a bounded number of times first; only blacklist if the
            # suggested wait is long enough to look like real daily-quota
            # exhaustion rather than a TPM cooldown.
            for attempt in range(4):
                try:
                    abstract = taa.get_abstract(problem)
                    break
                except Exception as e:
                    last_err = e
                    headers = getattr(getattr(e, "response", None), "headers", None)
                    retry_after = float(headers.get("retry-after", 0)) if headers else 0
                    is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                    if is_rate_limit and retry_after and retry_after <= 30 and attempt < 3:
                        time.sleep(retry_after + 1)
                        continue
                    if is_rate_limit:
                        exhausted_models.add(model)
                        print(f"  [model {model} now exhausted] (retry_after={retry_after})")
                    break
            if abstract:
                break

        if abstract:
            abstracts[name] = abstract
            mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
            fetch_status[name] = "found"
            done += 1
        else:
            print(f"  [{name}] all models failed/exhausted: {str(last_err)[:150]}")
            failed += 1

        if (i + 1) % 25 == 0:
            save(abstracts, ABSTRACT_CACHE_PATH)
            save(mech_emb, MECHANISM_EMBEDDING_CACHE_PATH)
            save(fetch_status, FETCH_STATUS_CACHE_PATH)
            print(f"  ...{i + 1}/{len(pending)} processed (done={done}, failed={failed})")

    save(abstracts, ABSTRACT_CACHE_PATH)
    save(mech_emb, MECHANISM_EMBEDDING_CACHE_PATH)
    save(fetch_status, FETCH_STATUS_CACHE_PATH)
    print(f"\nDone. {done} regenerated, {failed} still failed.")


if __name__ == "__main__":
    main()
