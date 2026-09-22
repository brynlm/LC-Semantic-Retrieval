"""
The prompt-instruction alone isn't reliably followed -- 3/12 regenerated
abstracts still leaked "stone"/"pile" even with explicit banned-word examples
added to SYSTEM_PROMPT. This retries generation for the still-leaking ones,
checking each attempt's mechanism text against the banned-word list and only
keeping a clean result, rather than trusting the model's compliance on faith.

Usage:
    python scratch_fix_remaining_leaks.py
"""

import pickle
import time

from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import test_approach_abstract as taa

MODEL_CHAIN = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-safeguard-20b"]
BANNED_WORDS = ["stone", "pile", "course", "prerequisite", "semester", "student", "coin",
                "ghost", "building", "cat", "mouse", "senate", "atom", "potion"]
TARGETS = ["parallel-courses", "parallel-courses-ii", "parallel-courses-iii"]
MAX_ATTEMPTS_PER_MODEL = 3


def leaks(mechanism: str) -> list[str]:
    text = mechanism.lower()
    return [w for w in BANNED_WORDS if w in text]


def get_clean_abstract(problem, exhausted_models):
    for model in MODEL_CHAIN:
        if model in exhausted_models:
            continue
        taa.MODEL = model
        for attempt in range(MAX_ATTEMPTS_PER_MODEL):
            try:
                abstract = taa.get_abstract(problem)
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
            found = leaks(abstract["mechanism"])
            if not found:
                return abstract, model, attempt
            print(f"    attempt with {model} still leaked {found}, retrying...")
    return None, None, None


def main():
    df = cs.load_dataset()
    name_to_code = dict(zip(df["name"], df["code"]))
    with open("abstract_discovery_cache.pkl", "rb") as f:
        abstracts = pickle.load(f)
    with open("mechanism_embedding_cache.pkl", "rb") as f:
        mech_emb = pickle.load(f)
    with open("official_editorial_code_cache.pkl", "rb") as f:
        official_code = pickle.load(f)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    exhausted_models = set()

    for name in TARGETS:
        code = official_code.get(name) or name_to_code[name]
        row = df[df["name"] == name].iloc[0]
        problem = {"name": name, "description": row["description"], "code": code}
        print(f"[{name}]")
        abstract, model, attempt = get_clean_abstract(problem, exhausted_models)
        if abstract:
            abstracts[name] = abstract
            mech_emb[name] = embed_model.encode(abstract["mechanism"], normalize_embeddings=True)
            print(f"  clean after model={model}, attempt={attempt}")
        else:
            print(f"  FAILED to get a clean abstract after exhausting retries")

    with open("abstract_discovery_cache.pkl", "wb") as f:
        pickle.dump(abstracts, f)
    with open("mechanism_embedding_cache.pkl", "wb") as f:
        pickle.dump(mech_emb, f)
    print("\nDone. Caches saved.")


if __name__ == "__main__":
    main()
