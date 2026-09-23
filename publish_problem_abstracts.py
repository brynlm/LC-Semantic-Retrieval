"""
Promotes reviewed entries from the staging caches
(problem_abstract_cache.staging.pkl / problem_embedding_cache.staging.pkl,
written by generate_problem_abstracts.py) into the production caches
(problem_abstract_cache.pkl / problem_embedding_cache.pkl) that a future
unified export_static_data.py will read.

This is the deliberate review gate between "generated" and "live": run
generate_problem_abstracts.py (possibly for hours, possibly over the whole
corpus) entirely without touching production, spot-check the staging
caches, then run this script to promote what's ready. Nothing here touches
merged_code_only_abstract_cache.pkl or static_site/data.json -- those
remain whatever the currently-deployed pipeline last produced until an
explicit, separate swap-over decision is made to point the site at these
new caches instead.

Idempotent: names already in production are skipped, so this can be run
repeatedly as staging accumulates over time (the ongoing "new problems keep
appearing" case) or once after a full-corpus regeneration (the "start over"
case) -- same script either way.

Usage:
    python publish_problem_abstracts.py
"""

import pickle


def load(path, default=None):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default if default is not None else {}


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    prod_abs = load("problem_abstract_cache.pkl")
    prod_emb = load("problem_embedding_cache.pkl")
    print(f"Production before publish: {len(prod_abs)} abstracts, {len(prod_emb)} embeddings")

    staged_abs = load("problem_abstract_cache.staging.pkl")
    staged_emb = load("problem_embedding_cache.staging.pkl")
    print(f"Staged: {len(staged_abs)} abstracts, {len(staged_emb)} embeddings")

    already_published = set(prod_abs) & set(staged_abs)
    if already_published:
        print(f"Skipping {len(already_published)} already-published problems.")

    n_published = 0
    for name, abstract in staged_abs.items():
        if name in already_published:
            continue
        prod_abs[name] = abstract
        prod_emb[name] = staged_emb[name]
        n_published += 1

    save(prod_abs, "problem_abstract_cache.pkl")
    save(prod_emb, "problem_embedding_cache.pkl")
    print(f"Published {n_published} new problems. Production after publish: "
          f"{len(prod_abs)} abstracts, {len(prod_emb)} embeddings")


if __name__ == "__main__":
    main()
