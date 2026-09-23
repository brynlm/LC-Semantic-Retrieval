"""
Generates abstracts for problems in problem_source_cache.pkl, using this
project's current default abstract-generation setup: fireworks/gpt-oss-120b
via Requesty, the canonical SYSTEM_PROMPT + response parser
(pipeline_lib._parse_analysis), and the leaks()/scrub_leaks()
narrative-leakage safety net.

Writes to STAGING caches, not the live production cache -- see
publish_problem_abstracts.py for the deliberate, separate step that
promotes reviewed entries into production. This lets a batch (including a
full from-scratch corpus regeneration) run for as long as it needs to
without ever touching what export_static_data.py / the deployed site
actually reads.

Two deliberate policies, applied uniformly to every problem regardless of
source:

1. Anonymization is conditional on is_python (from
   build_problem_source_cache.py). anonymize_code() is built on Python's
   own `ast` module and silently returns the ORIGINAL code on a parse error
   rather than raising -- so calling it on a non-Python entry would
   silently skip anonymization with no record that anything unusual
   happened. Instead: Python code is anonymized as usual; non-Python code
   is sent as-is, and every generated abstract records whether its input
   was anonymized. Same reasoning applies to
   pipeline_lib.is_usable_solution_code(), also ast.parse-based -- skipped
   for non-Python entries in favor of a simple non-empty/length sanity
   check.

2. Community solutions are not held to an execution-verified correctness
   bar (see fetch_problem_community_solutions.py's docstring for why --
   LeetCode alone doesn't provide a ready-made test harness). A
   highly-upvoted community solution is treated as a reasonable proxy for
   correctness without executing it. `verified` (from the source cache)
   still records which is which for anyone reviewing later.

Writes problem_abstract_cache.staging.pkl / problem_embedding_cache.staging.pkl.

Usage:
    python generate_problem_abstracts.py [--limit N]
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import pickle
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from pipeline_lib import SYSTEM_PROMPT, _parse_analysis, anonymize_code, is_usable_solution_code, leaks, scrub_leaks

load_dotenv()

HARD_TIMEOUT_S = 90
SOURCE_PATH = "problem_source_cache.pkl"
CACHE_PATH = "problem_abstract_cache.staging.pkl"
EMB_CACHE_PATH = "problem_embedding_cache.staging.pkl"
CHECKPOINT_EVERY = 20

MODEL = "fireworks/gpt-oss-120b"
MAX_TOKENS = 1500
MAX_RETRIES = 5
RETRY_DELAY_S = 5

_KEY = os.environ.get("REQUESTY_API_KEY")


def load(path, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _post(content):
    return requests.post(
        "https://router.requesty.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": "low",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        },
        timeout=60,
    )


def call_model(code: str):
    content = f"Solution code:\n{code}\n"
    for attempt in range(MAX_RETRIES):
        ex = ThreadPoolExecutor(max_workers=1)
        future = ex.submit(_post, content)
        try:
            r = future.result(timeout=HARD_TIMEOUT_S)
        except FutureTimeoutError:
            print(f"  hard timeout after {HARD_TIMEOUT_S}s (attempt {attempt + 1}), retrying")
            ex.shutdown(wait=False)
            continue
        except Exception as e:
            print(f"  request error (attempt {attempt + 1}): {type(e).__name__}: {str(e)[:150]}")
            ex.shutdown(wait=False)
            time.sleep(RETRY_DELAY_S)
            continue
        ex.shutdown(wait=False)

        if r.status_code == 200:
            data = r.json()
            if "choices" in data:
                text = (data["choices"][0]["message"]["content"] or "").strip()
                try:
                    abstract = _parse_analysis(text)
                except Exception as e:
                    print(f"  parse failed (attempt {attempt + 1}): {e}")
                    time.sleep(RETRY_DELAY_S)
                    continue
                leaked = leaks(abstract["mechanism"])
                if leaked:
                    abstract["mechanism"] = scrub_leaks(abstract["mechanism"], leaked)
                return abstract
        if r.status_code == 402:
            raise RuntimeError(f"Balance exhausted: {r.text[:200]}")
        print(f"  status={r.status_code} (attempt {attempt + 1}): {r.text[:200]}")
        time.sleep(RETRY_DELAY_S)
    return None


def main(limit=None):
    source = load(SOURCE_PATH, {})
    print(f"Source pool: {len(source)} problems")

    cache = load(CACHE_PATH, {})
    mech_emb = load(EMB_CACHE_PATH, {})
    print(f"Already generated (staging): {len(cache)}")

    remaining = [n for n in sorted(source) if n not in cache]
    if limit:
        remaining = remaining[:limit]
    print(f"Remaining to attempt: {len(remaining)}")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    n_done, n_failed, n_unusable = 0, 0, 0
    n_anonymized, n_not_anonymized = 0, 0
    stopped_early = False

    for i, name in enumerate(remaining):
        entry = source[name]
        code = entry["code"]
        is_python = entry["is_python"]

        if is_python:
            if not is_usable_solution_code(code):
                n_unusable += 1
                continue
            model_input = anonymize_code(code)
            anonymized = True
        else:
            if not code or len(code.strip()) < 20:
                n_unusable += 1
                continue
            model_input = code
            anonymized = False

        try:
            abstract = call_model(model_input)
        except RuntimeError as e:
            print(f"Stopping: {e}")
            stopped_early = True
            break
        except Exception as e:
            print(f"  [{name}] unexpected error, skipping: {type(e).__name__}: {str(e)[:150]}")
            abstract = None

        if abstract:
            abstract["source"] = entry["source"]
            abstract["anonymized"] = anonymized
            abstract["source_language"] = entry["language"]
            abstract["source_verified"] = entry["verified"]
            abstract["community_meta"] = entry.get("meta")
            cache[name] = abstract
            mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
            n_done += 1
            n_anonymized += anonymized
            n_not_anonymized += not anonymized
        else:
            n_failed += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save(cache, CACHE_PATH)
            save(mech_emb, EMB_CACHE_PATH)
            print(f"  ...{i + 1}/{len(remaining)} processed (done={n_done}, failed={n_failed}, "
                  f"unusable={n_unusable}, anonymized={n_anonymized}, not_anonymized={n_not_anonymized})")

    save(cache, CACHE_PATH)
    save(mech_emb, EMB_CACHE_PATH)
    pct = 100 * len(cache) / len(source) if source else 0
    print(f"\nDone{' (stopped early)' if stopped_early else ''}. "
          f"{n_done} new, {len(cache)} total staged ({pct:.1f}% of source pool), "
          f"{n_failed} failures, {n_unusable} unusable source skipped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
