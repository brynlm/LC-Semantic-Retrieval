"""
Quick single-example harness for testing an LLM "phase taxonomy" classifier:
splits a solution's technique into PREPROCESSING (steps that reorganize or
transform the input before the core logic runs) and ALGORITHM (the technique
that actually derives the answer from that structure), each a short list of
{label, role} entries.

This is step 1 of a discover -> canonicalize -> classify pipeline: labels are
free-generated here (no closed vocabulary yet) over a small, deliberately
varied hand-picked sample -- judge the preprocessing/algorithm boundary calls
and label quality by eye before doing anything like scaling to the full
dataset or canonicalizing a controlled vocabulary out of the results.

Mirrors test_approach_abstract.py's provider plumbing (groq/ollama/anthropic)
and tolerant-JSON-parse pattern -- see that file for provider setup notes
(API keys, `pip install` lines, etc).

Usage:
    python test_phase_taxonomy.py
"""

import json
import os

from dotenv import load_dotenv

import categorical_similarity as cs

load_dotenv()

PROVIDER = "groq"  # "ollama" | "groq" | "anthropic"
MODEL_BY_PROVIDER = {
    "ollama": "qwen2.5-coder:7b",
    "groq": "openai/gpt-oss-20b",
    "anthropic": "claude-sonnet-5",
}
MODEL = MODEL_BY_PROVIDER[PROVIDER]

SYSTEM_PROMPT = """You analyze competitive-programming solutions to classify the work they do \
into two ordered phases: PREPROCESSING (steps that reorganize or transform \
the input before the core logic runs) and ALGORITHM (the technique that \
actually derives the answer from that, possibly preprocessed, input).

Given a problem's description and a working solution, respond with ONLY a \
single JSON object (no markdown code fences, no commentary before or after) \
in exactly this shape:

{
  "preprocessing": [
    {
      "label": "<short, standard technique/data-structure name using widely-recognized algorithms terminology, e.g. \\"sorting\\", \\"prefix sum\\", \\"frequency count\\", \\"adjacency list construction\\", \\"coordinate compression\\">",
      "role": "<1 sentence on what this step organizes or transforms, in this solution>"
    }
  ],
  "algorithm": [
    {
      "label": "<short, standard technique/data-structure name, e.g. \\"two pointers\\", \\"binary search\\", \\"DFS\\", \\"monotonic stack\\", \\"0/1 knapsack DP\\", \\"union-find\\">",
      "role": "<1 sentence on how this technique derives the answer from the (possibly preprocessed) input, in this solution>"
    }
  ]
}

Rules:
- "preprocessing" holds steps that reorganize/transform the input before the \
  core technique runs and would still make sense done in isolation (e.g. \
  sorting an array, building a hash map, building a graph's adjacency list, \
  building a prefix-sum array). It is often empty -- do NOT invent a \
  preprocessing step if the solution has none, e.g. a single pass with a \
  running variable and no upfront transformation has no preprocessing.
- "algorithm" holds the technique(s) that actually derive the answer, \
  exploiting whatever structure is available (raw or preprocessed). Every \
  solution has at least one -- this list must not be empty.
- Classify by the role a technique plays in THIS solution, not some fixed \
  absolute category -- e.g. sorting is preprocessing when it merely enables \
  a later two-pointer scan, but IS the algorithm in a problem solved by \
  sorting alone.
- Use standard, widely-recognized algorithms/data-structure terminology (the \
  kind used in an algorithms textbook or competitive programming), not ad \
  hoc descriptions of what the code literally does line by line.
- List at most 2 entries per phase -- only genuinely distinct, load-bearing \
  techniques. Do not list implementation substeps as if they were separate \
  techniques (e.g. "index increment" or "array initialization" are not \
  techniques). If several steps are really one cohesive technique, describe \
  that as ONE entry.
- Do not name the specific problem or reuse phrasing copied from the problem \
  statement where avoidable.
- Output ONLY the JSON object. No markdown fences, no leading or trailing text.
"""

USER_TEMPLATE = """Problem description:
{description}

Solution code:
{code}
"""

# Hand-picked to stress-test the preprocessing/algorithm boundary, not to be
# representative -- each was chosen for a specific reason:
#   two-sum                  -- single hash-map pass, no preprocessing at all
#   3sum                     -- sorting (preprocessing) enables two pointers
#                                (algorithm): the clean, textbook two-phase case
#   merge-k-sorted-lists     -- is seeding the initial heap from list heads
#                                preprocessing, or part of the algorithm?
#   top-k-frequent-elements  -- frequency count (preprocessing) feeds a
#                                heap/bucket-sort (algorithm)
#   course-schedule          -- build adjacency list + in-degrees
#                                (preprocessing) feeds topological sort (algorithm)
#   number-of-islands        -- no preprocessing; DFS/BFS flood fill is the
#                                whole solution
SAMPLE_NAMES = [
    "two-sum",
    "3sum",
    "merge-k-sorted-lists",
    "top-k-frequent-elements",
    "course-schedule",
    "number-of-islands",
]


def _user_content(problem: dict) -> str:
    return USER_TEMPLATE.format(
        description=problem["description"].strip(),
        code=problem["code"].strip(),
    )


class TruncatedResponseError(RuntimeError):
    """Raised when a response was cut off by the token limit before the model
    finished writing -- see test_approach_abstract.py's identically-named
    class for the full rationale."""


def _parse_phase_taxonomy(raw_text: str) -> dict:
    """Same tolerant-JSON approach as test_approach_abstract._parse_analysis
    (strict json.loads first, json_repair fallback for almost-valid LLM
    output), validated against this script's {preprocessing, algorithm}
    schema instead."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {raw_text!r}")
    candidate = text[start : end + 1]

    try:
        analysis = json.loads(candidate)
    except json.JSONDecodeError as strict_err:
        try:
            import json_repair

            analysis = json_repair.loads(candidate)
        except Exception:
            raise ValueError(
                f"Model output wasn't valid JSON ({strict_err}), and "
                f"json_repair couldn't recover it either: {raw_text!r}"
            ) from strict_err
        if not isinstance(analysis, dict):
            raise ValueError(
                f"json_repair recovered non-dict output ({type(analysis)}): "
                f"{raw_text!r}"
            )
        print("  [warning: JSON needed repair -- worth tracking how often "
              "this happens across the full pool]")

    for phase in ("preprocessing", "algorithm"):
        entries = analysis.get(phase)
        if not isinstance(entries, list):
            raise ValueError(f"Missing/malformed '{phase}' list: {analysis!r}")
        for e in entries:
            if not isinstance(e, dict) or not e.get("label") or not e.get("role"):
                raise ValueError(f"Malformed {phase} entry: {e!r}")

    if not analysis["algorithm"]:
        raise ValueError(f"'algorithm' must not be empty: {analysis!r}")

    return analysis


def get_phase_taxonomy(problem: dict) -> dict:
    if PROVIDER == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_content(problem)}],
        )
        return _parse_phase_taxonomy(msg.content[0].text.strip())

    if PROVIDER == "groq":
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_content(problem)},
            ],
        )
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if choice.finish_reason == "length":
            reasoning = getattr(choice.message, "reasoning", None)
            raise TruncatedResponseError(
                f"Response truncated by max_tokens (reasoning_tokens="
                f"{resp.usage.completion_tokens_details.reasoning_tokens if resp.usage.completion_tokens_details else '?'}"
                f", content chars so far={len(content)}). Raise max_tokens, "
                f"lower reasoning_effort, or shrink the ask. Partial content: "
                f"{content[:200]!r} / partial reasoning: {str(reasoning)[:200]!r}"
            )
        return _parse_phase_taxonomy(content)

    if PROVIDER == "ollama":
        import ollama

        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_content(problem)},
            ],
        )
        return _parse_phase_taxonomy(resp["message"]["content"].strip())

    raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")


if __name__ == "__main__":
    df = cs.load_dataset()
    sample = df[df["name"].isin(SAMPLE_NAMES)]

    for _, problem in sample.iterrows():
        taxonomy = get_phase_taxonomy(problem[["name", "description", "code"]].to_dict())
        print(f"--- {problem['name']} ---")
        print(json.dumps(taxonomy, indent=2))
        print()
