"""
Regenerates abstracts using nvidia/nemotron-3-super-120b-a12b via Requesty
(https://router.requesty.ai), a $0-priced route that requires no account
balance. Chosen after Groq's pay-as-you-go signup was unavailable and this
model's output quality was spot-checked against 4 different problems,
matching (and in one case exceeding the precision of) the true gpt-oss-120b
results from the earlier Groq run -- e.g. correctly distinguishing "the two
odd-degree nodes are directly non-adjacent" from "a third node exists
non-adjacent to both" on add-edges-to-make-degrees-of-all-nodes-even, a
nuance the Groq 120b run's abstract for the same problem didn't spell out.

This model does substantial visible chain-of-thought reasoning before its
final JSON answer (no reasoning_effort control exposed on this route), so it
needs a much larger max_tokens budget than the Groq calls (4000, vs 1000
there) or it truncates before finishing -- confirmed by testing at 1000
first and seeing a mid-reasoning cutoff. Retries on Requesty's transient
"Service temporarily overloaded" 503s, which showed up on ~1 in 3 calls
during testing.

Writes to separate cache files (hq_abstract_cache_nemotron.pkl /
hq_mechanism_embedding_cache_nemotron.pkl) -- does not touch
hq_abstract_cache.pkl, hq_abstract_cache_120b.pkl, or any production cache.
Skips names already covered by the true-120b run (hq_abstract_cache_120b.pkl)
so this extends coverage rather than duplicating it.

Usage:
    python run_hq_abstract_generation_nemotron.py
"""

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import structural_features as sf
import test_approach_abstract as taa
from scratch_code_embedding_test import anonymize_code

load_dotenv()

# requests' own `timeout=` only bounds the gap between individual socket
# reads, not the total call duration -- if a proxy trickles occasional bytes
# to keep a slow/stuck connection alive, that per-read timeout never fires
# and the call can hang indefinitely (confirmed: one call hung for 3+ hours
# against a 60s requests timeout). A first attempt at fixing this with
# signal.alarm did NOT work -- SIGALRM doesn't reliably interrupt blocking
# I/O happening inside SSL/urllib3's C-level socket code, which can swallow
# the interrupt entirely (confirmed: hung again with zero timeout messages
# logged after 20+ minutes). Running the call in a worker thread and using
# .result(timeout=...) on the Future is more robust: it doesn't require the
# blocked call to cooperate at all -- if it never returns, we just stop
# waiting on it and move on. Deliberately a FRESH single-use executor per
# attempt (not one shared/reused pool) -- if a call truly hangs forever in
# its worker thread and we reused one persistent worker, every subsequent
# submission would queue behind that permanently-stuck thread and the whole
# script would wedge again, just one layer up from the original bug. A
# throwaway executor lets us abandon a stuck thread and keep going.
HARD_TIMEOUT_S = 90

HQ_SOURCE_PATH = "hq_source_code_cache.pkl"
ALREADY_120B_PATH = "hq_abstract_cache_120b.pkl"  # true gpt-oss-120b via Groq -- don't redo these
CACHE_PATH = "hq_abstract_cache_nemotron.pkl"
EMB_CACHE_PATH = "hq_mechanism_embedding_cache_nemotron.pkl"
CHECKPOINT_EVERY = 20

MODEL = "nvidia/nemotron-3-super-120b-a12b"
MAX_TOKENS = 4000
MAX_RETRIES = 6
RETRY_DELAY_S = 6

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
            "messages": [
                {"role": "system", "content": taa.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        },
        timeout=60,
    )


def call_nemotron(code: str):
    content = f"Solution code:\n{code}\n"
    for attempt in range(MAX_RETRIES):
        # NOT a context manager: `with ThreadPoolExecutor()` calls
        # shutdown(wait=True) on exit, which would block right here waiting
        # for a stuck thread to finish -- exactly the hang this is meant to
        # avoid. shutdown(wait=False) abandons a still-running thread
        # without waiting for it.
        ex = ThreadPoolExecutor(max_workers=1)
        future = ex.submit(_post, content)
        try:
            r = future.result(timeout=HARD_TIMEOUT_S)
        except FutureTimeoutError:
            print(f"  hard timeout after {HARD_TIMEOUT_S}s (attempt {attempt + 1}), abandoning and retrying")
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
                    return taa._parse_analysis(text)
                except Exception as e:
                    print(f"  parse failed (attempt {attempt + 1}): {e}")
                    time.sleep(RETRY_DELAY_S)
                    continue
        if r.status_code == 402:
            raise RuntimeError(f"Balance exhausted: {r.text[:200]}")
        print(f"  status={r.status_code} (attempt {attempt + 1}): {r.text[:200]}")
        time.sleep(RETRY_DELAY_S)
    return None


def main():
    hq_source = load(HQ_SOURCE_PATH, {})
    already_120b = load(ALREADY_120B_PATH, {})
    print(f"Higher-quality source pool: {len(hq_source)} problems")
    print(f"Already covered by true gpt-oss-120b (Groq): {len(already_120b)}")

    cache = load(CACHE_PATH, {})
    mech_emb = load(EMB_CACHE_PATH, {})
    print(f"Already generated (nemotron): {len(cache)}")

    remaining = [n for n in sorted(hq_source) if n not in cache and n not in already_120b]
    print(f"Remaining to attempt: {len(remaining)}")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    n_done, n_failed, n_unusable = 0, 0, 0
    stopped_early = False

    for i, name in enumerate(remaining):
        entry = hq_source[name]
        code = entry["code"]
        if not sf.is_usable_solution_code(code):
            n_unusable += 1
            continue
        anon = anonymize_code(code)

        try:
            abstract = call_nemotron(anon)
        except RuntimeError as e:
            print(f"Stopping: {e}")
            stopped_early = True
            break
        except Exception as e:
            # Anything unanticipated (network hiccup, unexpected response
            # shape, etc) -- log it and move on rather than losing the rest
            # of a multi-hour batch to one bad problem.
            print(f"  [{name}] unexpected error, skipping: {type(e).__name__}: {str(e)[:150]}")
            abstract = None

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
                  f"unusable={n_unusable}, total cached={len(cache)})")

    save(cache, CACHE_PATH)
    save(mech_emb, EMB_CACHE_PATH)
    pct = 100 * len(cache) / len(hq_source)
    print(f"\nDone{' (stopped early)' if stopped_early else ''}. "
          f"{n_done} new, {len(cache)} total cached ({pct:.1f}% of hq source pool), "
          f"{n_failed} failures, {n_unusable} unusable source skipped.")


if __name__ == "__main__":
    main()
