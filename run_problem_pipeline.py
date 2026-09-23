"""
Thin orchestrator for the HF-independent problem pipeline -- runs each stage
script's main() in sequence, stopping at the staging caches. Publishing
staged abstracts into production (publish_problem_abstracts.py) is a
deliberate, separate, manually-triggered step -- see its docstring for why.

LeetCode's live GraphQL API is the only source of truth throughout: no
frozen dataset snapshot is read anywhere in this chain. "Already have it"
always means "already in one of this pipeline's own cache files," so the
exact same code path handles picking up a handful of newly-published
problems and (given enough time and a high enough --limit) regenerating the
entire corpus from scratch -- there's no separate "bootstrap" mode.

Each stage remains independently runnable (`python <stage>.py`) and
independently resumable, since the LLM-generation and scraping stages are
rate-limited and checkpoint their own cache files as they go.

Stages, in order:
  1. discover_problems.py               -- refresh the full live LC listing
  2. fetch_problem_metadata.py           -- description/starter-code/tags
  3. fetch_problem_editorials.py         -- official editorial code
  4. fetch_problem_community_solutions.py -- community fallback
  5. build_problem_source_cache.py       -- reconcile to one code per problem
  6. generate_problem_abstracts.py       -- anonymize + LLM abstract + embed (staging)

`--limit N` caps how many pending problems stages 2/3/4/6 attempt this run
(discovery and reconciliation are whole-pool operations and aren't
meaningfully limited) -- pass a small N to smoke-test the whole chain
cheaply, or omit it for a full run (including, deliberately, a full
from-scratch corpus regeneration).

Usage:
    python run_problem_pipeline.py [--limit N] [--skip-discovery]
"""

import argparse

import build_problem_source_cache
import discover_problems
import fetch_problem_community_solutions
import fetch_problem_editorials
import fetch_problem_metadata
import generate_problem_abstracts


def main(limit=None, skip_discovery=False):
    def _stage(n, name):
        print(f"\n{'=' * 60}\nSTAGE {n}: {name}\n{'=' * 60}")

    if not skip_discovery:
        _stage(1, "discover_problems")
        discover_problems.main()
    else:
        print("\n(skipping stage 1: discover_problems, per --skip-discovery)")

    _stage(2, "fetch_problem_metadata")
    fetch_problem_metadata.main(limit=limit)

    _stage(3, "fetch_problem_editorials")
    fetch_problem_editorials.main(limit=limit)

    _stage(4, "fetch_problem_community_solutions")
    fetch_problem_community_solutions.main(limit=limit)

    _stage(5, "build_problem_source_cache")
    build_problem_source_cache.main()

    _stage(6, "generate_problem_abstracts")
    generate_problem_abstracts.main(limit=limit)

    print("\nPipeline complete (staging). Review problem_abstract_cache.staging.pkl, "
          "then run publish_problem_abstracts.py to promote it to production.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="cap pending problems processed by the fetch/generate stages (for smoke-testing)")
    ap.add_argument("--skip-discovery", action="store_true",
                     help="skip the live LC re-discovery pass and reuse the existing problem_catalog_cache.pkl")
    args = ap.parse_args()
    main(limit=args.limit, skip_discovery=args.skip_discovery)
