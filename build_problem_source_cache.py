"""
Reconciles the two code sources (official editorial, community fallback)
into one cache with a single winning code string per problem, editorial
preferred over community.

Records `language` and `is_python`, since non-editorial-Python solutions
exist in both sources (editorial java/cpp, community non-Python fallback).
`is_python` is what generate_problem_abstracts.py checks to decide whether
to run anonymize_code() at all: that function is built on Python's own
`ast` module and silently returns the ORIGINAL code on a parse error rather
than raising, so a non-Python solution passed to it would silently skip
anonymization with no signal that anything unusual happened. Recording
is_python explicitly here means that skip is a visible, deliberate decision
downstream, not a silent fallback.

Output: problem_source_cache.pkl -- {name: {"code": str, "source":
"editorial"|"community", "language": str, "is_python": bool,
"verified": bool, "meta": dict|None}}
`verified` is True only for editorial (LeetCode's own official solution);
every community entry is `verified=False` regardless of vote count --
"unverified" describes how the code was validated, not a claim about
whether it's likely correct.

Usage:
    python build_problem_source_cache.py
"""

import pickle


def load(path, default=None):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default if default is not None else {}


def main():
    editorial = load("problem_editorial_cache.pkl")
    community = load("problem_community_cache.pkl")
    metadata = load("problem_metadata_cache.pkl")

    all_slugs = sorted(set(metadata) & (set(editorial) | set(community)))

    reconciled = {}
    n_editorial, n_community, n_none = 0, 0, 0
    for slug in all_slugs:
        ed = editorial.get(slug, {})
        if ed.get("status") == "has_solution":
            reconciled[slug] = {
                "code": ed["code"],
                "source": "editorial",
                "language": ed["language"],
                "is_python": ed["language"] in ("python3", "python"),
                "verified": True,
                "meta": None,
            }
            n_editorial += 1
            continue

        cm = community.get(slug, {})
        if cm.get("status") in ("community_unverified", "community_unverified_nonpython"):
            reconciled[slug] = {
                "code": cm["code"],
                "source": "community",
                "language": cm["language"],
                "is_python": cm["language"] == "python",
                "verified": False,
                "meta": cm.get("meta"),
            }
            n_community += 1
            continue

        n_none += 1  # no editorial and no community solution found for this slug

    with open("problem_source_cache.pkl", "wb") as f:
        pickle.dump(reconciled, f)

    n_python = sum(1 for v in reconciled.values() if v["is_python"])
    n_nonpython = len(reconciled) - n_python
    print(f"problem_source_cache.pkl written: {len(reconciled)} problems")
    print(f"  from editorial (verified): {n_editorial}")
    print(f"  from community (unverified): {n_community}")
    print(f"  no code found: {n_none}")
    print(f"  python: {n_python}  non-python: {n_nonpython}")


if __name__ == "__main__":
    main()
