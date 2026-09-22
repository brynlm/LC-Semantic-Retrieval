"""
One-time reprocessing of new_problem_community_cache.pkl's
"community_unverified_nonpython" entries against run_new_problem_community_fetch's
now-fixed _as_usable_python (class-wrap retry) -- confirmed on the real
cached data that 10/26 were false negatives (valid Python whose community
post's code fence only had the class's methods, not the class declaration
itself), the other 16 genuinely Java/C++ or a truncated Python fragment
with a real syntax error (not something to auto-repair).

Operates entirely on already-fetched data -- no new LeetCode requests.

Usage:
    python fix_community_nonpython_falsepositives.py
"""

import pickle

from run_new_problem_community_fetch import _as_usable_python

CACHE_PATH = "new_problem_community_cache.pkl"


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    nonpy_slugs = [s for s, d in cache.items() if d["status"] == "community_unverified_nonpython"]
    print(f"Rechecking {len(nonpy_slugs)} community_unverified_nonpython entries...")

    recovered = []
    for slug in nonpy_slugs:
        d = cache[slug]
        usable = _as_usable_python(d["code"])
        if usable:
            cache[slug] = {"status": "community_unverified", "code": usable, "language": "python", "meta": d["meta"]}
            recovered.append(slug)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    print(f"Recovered {len(recovered)}/{len(nonpy_slugs)}: {recovered}")
    print(f"Remaining genuinely non-Python/unrecoverable: {len(nonpy_slugs) - len(recovered)}")


if __name__ == "__main__":
    main()
