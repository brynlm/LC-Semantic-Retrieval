"""
Regenerates abstracts for problems flagged by the narrative-leakage scan
(Stone Game family: "stone"/"pile"; Parallel Courses family: "courses") using
the strengthened SYSTEM_PROMPT (now explicitly names these as bad examples).
Keeps each problem's existing code source (official editorial if cached,
else dataset code) -- only the prompt/abstract changes, not the input.

Usage:
    python scratch_regenerate_leaky_abstracts.py
"""

import pickle
import time

from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import test_approach_abstract as taa

MODEL_CHAIN = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-safeguard-20b"]

TARGETS = [
    "stone-game", "stone-game-ii", "stone-game-iii", "stone-game-iv", "stone-game-v",
    "stone-game-vi", "stone-game-vii", "stone-game-viii", "stone-game-ix",
    "parallel-courses", "parallel-courses-ii", "parallel-courses-iii",
]


def get_abstract_with_backoff(problem, exhausted_models):
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
        old_mech = abstracts[name]["mechanism"]
        new_abstract = get_abstract_with_backoff(problem, exhausted_models)
        if not new_abstract:
            print(f"  [{name}] FAILED to regenerate")
            continue
        abstracts[name] = new_abstract
        mech_emb[name] = embed_model.encode(new_abstract["mechanism"], normalize_embeddings=True)
        leaked_before = any(w in old_mech.lower() for w in ("stone", "pile", "course"))
        leaked_after = any(w in new_abstract["mechanism"].lower() for w in ("stone", "pile", "course"))
        print(f"  [{name}] regenerated. leaked before={leaked_before}, leaked after={leaked_after}")

    with open("abstract_discovery_cache.pkl", "wb") as f:
        pickle.dump(abstracts, f)
    with open("mechanism_embedding_cache.pkl", "wb") as f:
        pickle.dump(mech_emb, f)
    print("\nDone. Caches saved.")


if __name__ == "__main__":
    main()
