"""
Tests generating mechanism abstracts from the solution code ALONE -- no
problem description in the prompt at all -- to see whether narrative leakage
(the model reaching for "stone"/"pile"/"course" in its own generated prose)
drops when the natural-language description isn't there to pull it toward
the problem's own flavor-text vocabulary. The code itself still contains the
original identifiers (e.g. `stoneGameVII`, `piles`) -- this tests whether the
model avoids echoing them into its own abstract without the description's
narrative pressure, not whether the code's own identifiers are clean.

Saves to separate caches (does not touch the production abstract/embedding
caches) so this stays a side-by-side comparison, not a silent pipeline swap.

Usage:
    python scratch_code_only_abstracts.py
"""

import csv
import os
import pickle
import re
import time

from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import test_approach_abstract as taa

MODEL_CHAIN = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-safeguard-20b"]
BANNED_WORDS = ["stone", "pile", "course", "prerequisite", "semester", "student", "coin",
                "ghost", "building", "cat", "mouse", "senate", "atom", "potion"]

CODE_ONLY_TEMPLATE = """Solution code:
{code}
"""

CACHE_PATH = "code_only_abstract_cache.pkl"
EMB_CACHE_PATH = "code_only_mechanism_embedding_cache.pkl"
CHECKPOINT_EVERY = 25


def leaks(mechanism: str) -> list[str]:
    """Whole-word match, plural-aware -- a naive substring check
    false-positives on "cat" in "concatenate", "pile" in "compile"/
    "compilation", "atom" in "atomic", "stone" in "milestone"/"keystone",
    "course" in "recourse"/"discourse", etc., which wastes retries (and
    rate-limit budget) rejecting abstracts that never actually leaked
    anything -- but a bare \\bword\\b also misses the plural ("stones"),
    which a first pass of this script let through uncaught (stone-game-ii's
    cached abstract said "the total remaining stones" and passed anyway)."""
    text = mechanism.lower()
    return [w for w in BANNED_WORDS if re.search(rf"\b{re.escape(w)}e?s?\b", text)]


def get_code_only_abstract(code, exhausted_models):
    content = CODE_ONLY_TEMPLATE.format(code=code.strip())
    for model in MODEL_CHAIN:
        if model in exhausted_models:
            continue
        taa.MODEL = model
        for attempt in range(2):
            try:
                from groq import Groq
                client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=0)
                kwargs = dict(
                    model=model,
                    max_tokens=1000,
                    messages=[
                        {"role": "system", "content": taa.SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                )
                try:
                    resp = client.chat.completions.create(reasoning_effort="low", **kwargs)
                except Exception as e:
                    if "reasoning_effort" in str(e):
                        resp = client.chat.completions.create(**kwargs)
                    else:
                        raise
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                abstract = taa._parse_analysis(text)
            except Exception as e:
                headers = getattr(getattr(e, "response", None), "headers", None)
                retry_after = float(headers.get("retry-after", 0)) if headers else 0
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                if is_rate_limit and retry_after and retry_after <= 30:
                    time.sleep(retry_after + 1)
                    continue
                if is_rate_limit:
                    exhausted_models.add(model)
                    break
                continue
            if not leaks(abstract["mechanism"]):
                return abstract
    return None


def main():
    df = cs.load_dataset()
    name_to_code = dict(zip(df["name"], df["code"]))
    with open("official_editorial_code_cache.pkl", "rb") as f:
        official_code = pickle.load(f)

    with open("/tmp/audit_verdicts.csv") as fcsv:
        audit_rows = list(csv.DictReader(fcsv))
    seen = set()
    dedup = []
    for r in audit_rows:
        if r["query"] in seen:
            continue
        seen.add(r["query"])
        dedup.append(r)

    names_needed = set()
    for r in dedup:
        names_needed.add(r["query"])
        names_needed.add(r["neighbor"])
    names_needed &= set(name_to_code)
    names_list = sorted(names_needed)

    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    mech_emb = {}
    if os.path.exists(EMB_CACHE_PATH):
        with open(EMB_CACHE_PATH, "rb") as f:
            mech_emb = pickle.load(f)

    exhausted_models = set()
    n_done, n_failed = 0, 0
    for i, name in enumerate(names_list):
        if name in cache:
            continue
        code = official_code.get(name, name_to_code[name])
        abstract = get_code_only_abstract(code, exhausted_models)
        if abstract:
            cache[name] = abstract
            mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
            n_done += 1
        else:
            n_failed += 1
            print(f"  [{name}] failed / still leaked after retries")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
            with open(EMB_CACHE_PATH, "wb") as f:
                pickle.dump(mech_emb, f)
            print(f"  ...{i + 1}/{len(names_list)} processed (done={n_done}, failed={n_failed})")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    with open(EMB_CACHE_PATH, "wb") as f:
        pickle.dump(mech_emb, f)
    print(f"\nDone. {n_done} new, {len(cache)} total cached, {n_failed} failures.")


if __name__ == "__main__":
    main()
