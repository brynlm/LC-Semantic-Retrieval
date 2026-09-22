import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd

import categorical_similarity as cs
import evaluate_categorical_similarity as ev

df = cs.load_dataset()
with open("abstract_discovery_cache.pkl", "rb") as f:
    abstracts = pickle.load(f)
with open("mechanism_embedding_cache.pkl", "rb") as f:
    mechanism_embeddings = pickle.load(f)

pool = sorted(set(abstracts) & set(mechanism_embeddings) & set(df["name"]))
names = pool
vectors = np.stack([mechanism_embeddings[n] for n in names])
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
cos_sim = vectors @ vectors.T
mechanism_sim = pd.DataFrame(cos_sim, index=names, columns=names)

legitimacy = ev.load_ground_truth_legitimacy()
pool_set = set(pool)
legit_pairs = []
for pair, label in legitimacy.items():
    if label != "LEGITIMATE":
        continue
    a, b = tuple(pair)
    if a in pool_set and b in pool_set:
        legit_pairs.append((a, b))

print(f"LEGITIMATE pairs fully in pool: {len(legit_pairs)}")

# --- generic-title-word stoplist, derived from corpus-wide frequency ---
GENERIC_EXTRA = {
    "of", "the", "a", "an", "and", "or", "with", "to", "in", "for", "from", "by",
    "at", "as", "is", "are", "your", "you", "i", "ii", "iii", "iv", "v", "vi",
}
token_counts = Counter()
title_tokens = {}
for n in pool:
    toks = [t for t in n.split("-") if t]
    title_tokens[n] = toks
    token_counts.update(set(toks))

GENERIC_THRESHOLD = 8  # a title-token appearing in >= this many titles is "generic vocabulary"
generic_words = GENERIC_EXTRA | {t for t, c in token_counts.items() if c >= GENERIC_THRESHOLD}


def domain_terms(name):
    return [t for t in title_tokens[name] if t not in generic_words and len(t) > 2 and not t.isdigit()]


def self_leaks(name):
    mech = abstracts.get(name, {}).get("mechanism", "").lower()
    terms = domain_terms(name)
    hits = [t for t in terms if re.search(r"\b" + re.escape(t) + r"\b", mech)]
    return hits


# overall leakage rate across the whole classified pool
leak_count = 0
no_domain_terms = 0
for n in pool:
    terms = domain_terms(n)
    if not terms:
        no_domain_terms += 1
        continue
    if self_leaks(n):
        leak_count += 1
evaluable_for_leakage = len(pool) - no_domain_terms
print(f"\nPool-wide literal title-noun leakage: {leak_count}/{evaluable_for_leakage} "
      f"({leak_count/evaluable_for_leakage:.1%}) of problems with a distinctive domain noun in title "
      f"leak that noun into their own mechanism text.")
print(f"({no_domain_terms} problems had no non-generic title words to check.)")

# --- rank of partner for every LEGITIMATE pair ---
k_pool = len(pool) - 1
rows = []
for a, b in legit_pairs:
    rank_ab = mechanism_sim[a].drop(index=a).rank(ascending=False)[b]
    rank_ba = mechanism_sim[b].drop(index=b).rank(ascending=False)[a]
    worst_rank = max(rank_ab, rank_ba)
    a_leaks = bool(self_leaks(a))
    b_leaks = bool(self_leaks(b))
    rows.append({"a": a, "b": b, "rank_ab": rank_ab, "rank_ba": rank_ba, "worst_rank": worst_rank,
                  "a_leaks": a_leaks, "b_leaks": b_leaks})

rank_df = pd.DataFrame(rows)
print(f"\nOf {len(rank_df)} LEGITIMATE pairs (pool size {len(pool)}):")
print(f"  worst_rank <= 10:  {(rank_df.worst_rank <= 10).sum()} ({(rank_df.worst_rank <= 10).mean():.1%})")
print(f"  worst_rank 11-20:  {((rank_df.worst_rank > 10) & (rank_df.worst_rank <= 20)).sum()} "
      f"({((rank_df.worst_rank > 10) & (rank_df.worst_rank <= 20)).mean():.1%})")
print(f"  worst_rank > 20:   {(rank_df.worst_rank > 20).sum()} ({(rank_df.worst_rank > 20).mean():.1%})")

poor = rank_df[rank_df.worst_rank > 20].copy()
poor["either_leaks"] = poor.a_leaks | poor.b_leaks
poor["neither_leaks"] = ~poor.a_leaks & ~poor.b_leaks
print(f"\nOf the {len(poor)} poorly-ranked (>20) LEGITIMATE pairs:")
print(f"  at least one side has literal title-noun leakage: {poor.either_leaks.sum()} ({poor.either_leaks.mean():.1%})")
print(f"  NEITHER side leaks literal title nouns (rule technically followed, still poor rank): "
      f"{poor.neither_leaks.sum()} ({poor.neither_leaks.mean():.1%})")

poor_sorted = poor.sort_values("worst_rank", ascending=False)
print("\n--- Worst-ranked LEGITIMATE pairs (rule-adherence noted) ---")
for _, r in poor_sorted.iterrows():
    print(f"{r.a} <-> {r.b}  worst_rank={int(r.worst_rank)}  a_leaks={r.a_leaks} b_leaks={r.b_leaks}")

with open("/tmp/poor_legit_pairs.pkl", "wb") as f:
    pickle.dump(poor_sorted, f)
