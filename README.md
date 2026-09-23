# LC Semantic Retrieval

**Live: https://lc-semantic-retrieval.lc-semantic-retrieval.workers.dev**

Find LeetCode problems by describing the *solution mechanism* you're thinking of ("two
pointers closing in from both ends while comparing endpoint sums", "monotonic stack for
next-greater-element") rather than by keyword. Search runs entirely in the browser — no
server holds an index or computes a similarity score.

## Why mechanism, not keywords

LeetCode's own tags and title text describe *what domain* a problem is in ("Array",
"Two Pointers"), not *how* the solution actually works. Two problems can share every tag
and still be solved by unrelated techniques, or share no tags and be near-identical in
mechanism. So instead of matching on tags or problem text, each problem is described by an
LLM-generated abstract of its solution's mechanism — written in a fixed, technique-first
register ("the two-pointer scan converges toward the middle while comparing endpoint
sums", not "solves the two-sum-like problem") — and everything is embedded and compared
in that space instead.

A live search query goes through the same two steps in reverse: an LLM rewrites the raw
query into that same register, then the browser embeds it with the same model used to
build the corpus, so a casually-phrased query and a formally-written abstract end up
comparable.

## Architecture

```
offline pipeline (this repo, run locally)          →  static_site/data.json
  discover → fetch solution (editorial, community
  fallback) → anonymize → generate LLM abstract of
  the *mechanism* → embed → publish

              ┌─────────────────────────────────────────────┐
              │            Cloudflare Worker                │
  query  ───▶  │  /api/rewrite: rewrite free text into the   │
  text        │  corpus's register (Groq LLM chain)          │
              │  everything else: serve static_site/ as-is   │
              └─────────────────────────────────────────────┘
                              │
                              ▼
              browser: embed the rewritten query with the
              same model (transformers.js, in-browser, no
              server round-trip) → cosine similarity against
              every precomputed embedding in data.json
```

The only server-side compute is the one-sentence query rewrite. Everything else —
loading the embedding model, embedding the query, ranking 3,400+ problems — happens
client-side, so the whole thing runs on Cloudflare's free tier: a static asset host plus
one cheap LLM call per search.

### Offline pipeline (root of this repo)

Every field in the corpus — description, solution code, mechanism abstract, embedding —
comes from one pipeline sourced entirely from LeetCode's own GraphQL API.
1. **Discover** (`discover_problems.py`) — paginates LeetCode's live problem list,
   scoped to the `algorithms` category (SQL/pandas/JS-only problems don't fit this
   project's model even when superficially Python-shaped).
2. **Fetch metadata** (`fetch_problem_metadata.py`) — description, starter code, tags,
   difficulty, for every free problem not yet cached.
3. **Fetch a solution** (`fetch_problem_editorials.py`, then
   `fetch_problem_community_solutions.py` as a fallback) — the official editorial where
   available, otherwise a vetted community solution. Provenance is tracked explicitly
   throughout: verified/editorial and unverified/community are never conflated.
4. **Reconcile** (`build_problem_source_cache.py`) — editorial preferred over community,
   down to one winning code string per problem.
5. **Anonymize the code** (`pipeline_lib.anonymize_code`) — an AST pass renames every
   local variable, parameter, and function name to a placeholder (`v0`, `v1`, ...) before
   the code ever reaches an LLM.
6. **Generate a mechanism abstract + embed** (`generate_problem_abstracts.py`) — an LLM
   describes *how* the anonymized solution works, in a fixed technique-first register,
   independent of the problem's domain framing, then embeds it with
   `sentence-transformers` (`all-MiniLM-L6-v2`, matched to the browser's
   `transformers.js` build of the same model so client-computed query embeddings land in
   the same space). Writes to a staging cache, not production.
7. **Publish** (`publish_problem_abstracts.py`) — promotes staged entries into this
   pipeline's own production cache. Kept as a deliberate, separate step rather than
   folded into generation, so a batch (including a full corpus regeneration, which can
   run for hours) can be spot-checked before it's able to affect anything downstream —
   including, later, the cache `export_static_data.py` actually reads (see below).

**Why anonymize the code first:** an LLM asked to describe a solution's mechanism will
happily lean on variable names instead — a solution to a course-scheduling problem with
variables named `prerequisites`/`courses` gets described in terms of courses and
prerequisites, even though the actual mechanism (cycle detection via topological sort)
applies just as well to scheduling tasks, ordering dependencies, or any other directed
graph. Renaming those to `v0`/`v1` before the code reaches the LLM forces the abstract to
describe the graph/traversal mechanism itself rather than reflecting back the domain the
variable names happened to suggest — confirmed with a controlled A/B test: de-anonymized
code measurably caused *category-convergence*, where abstracts drifted toward resembling
their surface domain rather than their actual mechanism.

### Static site (`static_site/`)

Plain HTML/JS, no build step. `data.json` ships each problem's tags, technique
breakdown, and precomputed embedding — deliberately **not** solution code (see
`export_static_data.py`'s docstring: the raw dataset's displayed code was sometimes a
different implementation than what the abstract was actually generated from, and
redistributing editorial/community solution text raises its own questions independent of
that). The site links out to each problem's own LeetCode page instead.

### Worker (`worker/index.js`)

Two responsibilities, nothing else: proxy `/api/rewrite` to a chain of Groq models (falls
back through the chain on failure, returns `null` rather than an error if every model
fails — the client treats "no rewrite available" as a graceful degradation, not a bug),
and an anonymized visit counter (`cf-connecting-ip` hashed with a private salt before
ever touching storage, so raw IPs are never persisted — see the comment above
`trackVisit` for the exact scope: it fires once per page load, not per asset or API
call).

## Repo layout

- **root `*.py`** (`discover_problems.py` through `publish_problem_abstracts.py`, plus
  `export_static_data.py`) — the pipeline described above, sourced entirely from
  LeetCode's own API.
- **`pipeline_lib.py`** — shared building blocks: AST-based code anonymization, the
  narrative-leakage safety net, and the abstract-generation prompt/parser contract.
- **`experiments/`** — one-off exploration and already-applied fix scripts: the specific
  batch-generation runs (e.g. the 120B/Requesty run that produced the deployed abstracts),
  quality-audit scratch work, an AST-based structural-feature/motif-detection side
  experiment that fed a lightweight solution-usability filter into the batch scripts but
  was never fed to the LLM itself, alternate model trials, and taxonomy discovery. Kept
  for methodology transparency, not meant to be re-run as-is.
- **`static_site/`** — the deployed frontend.
- **`worker/`** — the Cloudflare Worker source.
- **`manual_quality_audit_verdicts.csv`** — hand-verified match/partial/no-match verdicts
  used to sanity-check retrieval quality independent of any single automated metric.

## On the LeetCode scraping scripts

This repo includes the scripts used to fetch problem metadata and editorial/community
solution code from LeetCode, for transparency about methodology. **No scraped solution
code or full problem text is redistributed anywhere in this repo or the deployed site** —
`data.json` ships only tags, a technique breakdown, and embeddings; the deployed site
links to each problem's own LeetCode page rather than displaying its code. Local data
caches (`*.pkl`) are gitignored and never committed.

