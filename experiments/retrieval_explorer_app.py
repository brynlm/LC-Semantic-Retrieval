"""
Local interactive tool for manually auditing the code-only similarity
signal: start from a problem, see its top-K neighbors under a blend of
code-only mechanism-abstract embeddings and tag-based similarity, click any
neighbor to re-center the graph walk on it, and adjust K / the blend weight
live. Built after concluding the hand-tailored structural-feature-extractor
detour (REPAIRS/GROWS/etc.) wasn't scaling well -- this goes back to
auditing the two signals that already exist and are trusted
(code_only_vs_production_verdict.md), rather than continuing to hand-add
detectors.

Both similarity matrices are precomputed once at startup (639 problems is
small enough to hold every pairwise score in memory: two 639x639 float32
matrices, ~3.3MB combined) so every request is a simple slice + blend, no
per-request recomputation.

Also serves /api/query: a free-text prompt ("something with two pointers
closing in from both ends") gets rewritten by an LLM into the same register
the corpus's own mechanism abstracts use (query_rewrite.py), embedded with
the same model, and matched against the same emb_matrix -- the one genuinely
new LLM-in-the-loop feature, versus embedding the raw prompt directly.

Usage:
    python retrieval_explorer_app.py
    (then open http://127.0.0.1:5001 )
"""

import pickle

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sentence_transformers import SentenceTransformer

import categorical_similarity as cs
import query_rewrite as qr

app = Flask(__name__)

# same model the corpus's own mechanism abstracts were embedded with -- a
# live query MUST be embedded with this exact model to land in a comparable
# region of the same 384-dim space as emb_matrix below.
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

df = cs.load_dataset()

with open("merged_code_only_abstract_cache.pkl", "rb") as f:
    ABSTRACTS = pickle.load(f)
with open("merged_code_only_mechanism_embedding_cache.pkl", "rb") as f:
    EMBEDDINGS = pickle.load(f)

POOL = sorted(set(ABSTRACTS) & set(EMBEDDINGS) & set(df["name"]))
NAME_TO_IDX = {name: i for i, name in enumerate(POOL)}

pool_df = df[df["name"].isin(set(POOL))].set_index("name").loc[POOL].reset_index()

emb_matrix = np.stack([EMBEDDINGS[n] for n in POOL]).astype(np.float32)
emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
CODE_SIM = emb_matrix @ emb_matrix.T

TAG_SIM = cs.tag_similarity_matrix(pool_df, idf_weighted=True, metric="dice").loc[POOL, POOL].to_numpy()

DATA_BY_NAME = {}
for _, row in pool_df.iterrows():
    DATA_BY_NAME[row["name"]] = {
        "name": row["name"],
        "difficulty": row["difficulty"],
        "tags": list(row["tags"]),
        "description": row["description"],
        "code": row["code"],
        "abstract": ABSTRACTS[row["name"]],
    }


@app.route("/")
def index():
    return send_from_directory(".", "retrieval_explorer.html")


@app.route("/api/problems")
def api_problems():
    q = request.args.get("q", "").strip().lower()
    names = POOL if not q else [n for n in POOL if q in n.lower()]
    return jsonify(names[:50])


@app.route("/api/problem/<path:name>")
def api_problem(name):
    if name not in DATA_BY_NAME:
        return jsonify({"error": "not found"}), 404
    return jsonify(DATA_BY_NAME[name])


@app.route("/api/neighbors/<path:name>")
def api_neighbors(name):
    if name not in NAME_TO_IDX:
        return jsonify({"error": "not found"}), 404
    k = max(1, min(int(request.args.get("k", 10)), len(POOL) - 1))
    alpha = max(0.0, min(float(request.args.get("alpha", 0.7)), 1.0))
    idx = NAME_TO_IDX[name]

    code_scores = CODE_SIM[idx]
    tag_scores = TAG_SIM[idx]
    combined = alpha * code_scores + (1 - alpha) * tag_scores

    order = np.argsort(-combined)
    results = []
    for i in order:
        other = POOL[i]
        if other == name:
            continue
        entry = DATA_BY_NAME[other]
        results.append({
            "name": other,
            "combined_score": float(combined[i]),
            "code_score": float(code_scores[i]),
            "tag_score": float(tag_scores[i]),
            "difficulty": entry["difficulty"],
            "tags": entry["tags"],
            "mechanism": entry["abstract"].get("mechanism", ""),
        })
        if len(results) >= k:
            break
    return jsonify(results)


@app.route("/api/query")
def api_query():
    prompt = request.args.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "empty prompt"}), 400
    k = max(1, min(int(request.args.get("k", 10)), len(POOL)))

    rewritten = qr.rewrite_query(prompt)
    if rewritten is None:
        return jsonify({"error": "query rewrite failed -- all models unavailable, try again"}), 502

    query_vec = EMBED_MODEL.encode(rewritten, normalize_embeddings=True).astype(np.float32)
    scores = emb_matrix @ query_vec  # cosine similarity: both sides are L2-normalized

    order = np.argsort(-scores)[:k]
    results = []
    for i in order:
        name = POOL[i]
        entry = DATA_BY_NAME[name]
        results.append({
            "name": name,
            "score": float(scores[i]),
            "difficulty": entry["difficulty"],
            "tags": entry["tags"],
            "mechanism": entry["abstract"].get("mechanism", ""),
        })
    return jsonify({"rewritten": rewritten, "results": results})


if __name__ == "__main__":
    print(f"Pool size: {len(POOL)}")
    app.run(debug=True, port=5001)
