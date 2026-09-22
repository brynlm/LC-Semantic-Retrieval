"""
Completes the hq_abstract_cache_120b.pkl pool (currently 54/2384, generated
via Groq before its daily 120b quota wall) by routing the same model,
gpt-oss-120b, through Requesty's paid fireworks/gpt-oss-120b endpoint
instead -- confirmed reliable and cheap ($0.15/1M input, $0.60/1M output;
~$1-2 for the full remaining pool) throughout this session's testing.

Deliberately keeps anonymization ON: the model-upgrade win (fixing
best-time-to-buy-and-sell-stock-iii's dropped second transaction phase) was
isolated to model size alone across all four test variants, all of which
used anonymized code -- so it doesn't require dropping anonymization to
capture. Dropping anonymization was separately shown to buy only a partial
fix on one case (-with-cooldown) while introducing confirmed, reproducible
harm elsewhere (Stone Game family internal crowding via category-recognition
convergence, cat-and-mouse-ii/escape-the-ghosts spurious inflation) that
isn't fixable by leak-word detection since no banned words were involved.
Given a confirmed-but-partial upside against confirmed downside, this run
stays anonymized and banks the clean, already-validated win.

Still applies leaks()/scrub_leaks() as a safety net -- confirmed even
anonymized code can trip a leak occasionally (parallel-courses-ii leaked
"prerequisite" even anonymized, a memorized-idiom effect unrelated to
reading actual identifiers), so this costs nothing to keep on.

Writes into the SAME hq_abstract_cache_120b.pkl / hq_mechanism_embedding_cache_120b.pkl
used by the original Groq-based run -- treated as one unified cache since
it's the same underlying model, just a different host.

Usage:
    python run_hq_abstract_generation_120b_requesty.py
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
from scratch_code_only_abstracts import leaks, scrub_leaks

load_dotenv()

HARD_TIMEOUT_S = 90  # see run_hq_abstract_generation_nemotron.py docstring for why
                      # this uses a fresh ThreadPoolExecutor per attempt rather than
                      # relying on requests' own timeout or signal.alarm

HQ_SOURCE_PATH = "hq_source_code_cache.pkl"
ALREADY_NEMOTRON_PATH = "hq_abstract_cache_nemotron.pkl"  # different model -- don't redo, keep separate
CACHE_PATH = "hq_abstract_cache_120b.pkl"
EMB_CACHE_PATH = "hq_mechanism_embedding_cache_120b.pkl"
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
                {"role": "system", "content": taa.SYSTEM_PROMPT},
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
                    abstract = taa._parse_analysis(text)
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


def main():
    hq_source = load(HQ_SOURCE_PATH, {})
    already_nemotron = load(ALREADY_NEMOTRON_PATH, {})
    print(f"Higher-quality source pool: {len(hq_source)} problems")
    print(f"Already covered by nemotron (separate model, skip): {len(already_nemotron)}")

    cache = load(CACHE_PATH, {})
    mech_emb = load(EMB_CACHE_PATH, {})
    print(f"Already generated (120b, this cache): {len(cache)}")

    remaining = [n for n in sorted(hq_source) if n not in cache and n not in already_nemotron]
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
            abstract = call_model(anon)
        except RuntimeError as e:
            print(f"Stopping: {e}")
            stopped_early = True
            break
        except Exception as e:
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
