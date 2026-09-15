"""
Reconciles the two higher-quality code sources (official editorial,
verified community solution) into one cache with a single winning code
string per problem, editorial preferred over community, tagged with
provenance. This is the only thing the abstract-generation step reads --
the two raw per-source caches stay untouched and separate.

Output: hq_source_code_cache.pkl -- {name: {"code": str, "source":
"editorial"|"community", "meta": dict|None}}

Usage:
    python build_hq_source_cache.py
"""

import pickle


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    editorial_code = load("official_editorial_code_cache.pkl")
    community_status = load("community_verified_status_cache.pkl")
    community_code = load("community_verified_solution_cache.pkl")
    community_meta = load("community_verified_meta_cache.pkl")

    verified_community_names = {n for n, s in community_status.items() if s == "verified"}

    hq = {}
    for name, code in editorial_code.items():
        hq[name] = {"code": code, "source": "editorial", "meta": None}

    added_from_community = 0
    for name in verified_community_names:
        if name in hq:
            continue  # editorial already wins
        hq[name] = {"code": community_code[name], "source": "community", "meta": community_meta.get(name)}
        added_from_community += 1

    with open("hq_source_code_cache.pkl", "wb") as f:
        pickle.dump(hq, f)

    n_editorial = sum(1 for v in hq.values() if v["source"] == "editorial")
    n_community = sum(1 for v in hq.values() if v["source"] == "community")
    print(f"hq_source_code_cache.pkl written: {len(hq)} problems total")
    print(f"  from editorial: {n_editorial}")
    print(f"  from community (editorial had none): {n_community}")


if __name__ == "__main__":
    main()
