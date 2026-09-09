"""
Quick single-example harness for testing LLM-generated "approach abstracts."

Goal: given a problem description + a working solution, produce a short,
normalized description of the *algorithmic technique* -- stripped of
problem-specific framing -- that you can later embed and compare across
problems.

Run against a couple of hand-picked problems where you already know the
"right" answer intuitively, and judge the output by eye before scaling up.

Provider options (pick one via PROVIDER below -- no paid API required):

  "ollama"  Free, fully local. Install from ollama.com, then e.g.:
                ollama pull qwen2.5-coder:7b
            No API key needed.
                pip install ollama --break-system-packages

  "groq"    Free-tier cloud API serving open-weight models (Llama 3.3 70B,
            GPT-OSS 20B/120B, etc) very fast. Get a free key at
            console.groq.com, then:
                export GROQ_API_KEY=...
                pip install groq --break-system-packages

  "anthropic"  Paid API (not covered by a Claude Code/Pro/Max subscription --
            billed separately at API rates). Kept here for comparing output
            quality against the free options once you've picked one.
                export ANTHROPIC_API_KEY=...
                pip install anthropic --break-system-packages

Usage:
    python test_approach_abstract.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

PROVIDER = "groq"  # "ollama" | "groq" | "anthropic"

# Model to use for the chosen provider. Check current best small coding
# model at ollama.com/library (Ollama) or console.groq.com/docs/models (Groq)
# -- these lists change often.
MODEL_BY_PROVIDER = {
    "ollama": "qwen2.5-coder:7b",
    "groq": "openai/gpt-oss-20b",
    "anthropic": "claude-sonnet-5",
}
MODEL = MODEL_BY_PROVIDER[PROVIDER]

SYSTEM_PROMPT = """You analyze competitive-programming solutions to extract their \
algorithmic structure: the individual techniques involved, and the mechanism \
that ties them together.
 
Given a problem's description and a working solution, respond with ONLY a \
single JSON object (no markdown code fences, no commentary before or after) \
in exactly this shape:
 
{
  "techniques": [
    {
      "name": "<short technique/data-structure name, e.g. \\"heap\\", \\"hash set\\", \\"prefix sum\\">",
      "role": "<1 sentence, in plain language, on what this technique accomplishes in this solution. For niche or compound techniques, explain the underlying mechanism itself rather than just naming it -- e.g. for lazy deletion from a heap: \\"stale entries are left in place and skipped over lazily when they surface at the top, rather than being removed immediately on invalidation.\\">"
    }
  ],
  "mechanism": "<2-4 sentences describing how the techniques above work together to solve the problem. Write this around the underlying goal/invariant (e.g. maintaining a running target sum, finding the nearest valid boundary) rather than leading with any single technique's name. This field must stand alone and read clearly without the techniques list for context -- it is what gets embedded for similarity search.>"
}
 
Rules:
- List at most 3 techniques -- the ones that are genuinely distinct and \
  load-bearing for the solution. Do not rank or single out a "primary" one, \
  but do NOT list implementation substeps as if they were separate \
  techniques (e.g. "index increment" or "variable initialization" are not \
  techniques). If several steps are really just parts of one cohesive \
  technique (e.g. scanning a string character-by-character while \
  maintaining running state), describe that as ONE technique, not several.
- Translate every problem-specific noun into the bare computer-science object \
  it actually is, and write the whole "mechanism" field using ONLY that \
  vocabulary -- "graph", "node", "directed edge", "interval", "array", \
  "sequence", "tree", "set", "state", "connected component", and so on. Never \
  use the problem's own nouns (course, semester, student, letter, coin, ghost, \
  building, cat, mouse, senate, atom, potion, stone, pile, prerequisite) or \
  thin rewordings of them -- e.g. "tasks with prerequisite dependencies" is \
  still the problem's own narrative dressed up; the actual CS object is \
  "nodes with precedence edges." This applies even when a problem is one of \
  a same-title series (e.g. "Stone Game", "Parallel Courses", "Course \
  Schedule" I/II/III...) -- reusing that series' flavor noun (stone, pile, \
  course) makes every entry in the series look artificially similar to every \
  other regardless of whether their actual techniques match, which is \
  exactly the failure mode this rule exists to prevent. Two problems that \
  apply the same technique to the same kind of object should read in \
  near-identical CS vocabulary even when their surface stories have nothing \
  in common. For example:
    BAD:  "tasks that must be completed in an order respecting prerequisite \
          dependencies between them"
    GOOD: "nodes in a directed graph that must be ordered so every edge \
          points from an earlier node to a later one"
    BAD:  "the number of ways to climb a staircase by advancing one or two \
          steps at a time"
    GOOD: "the number of ways to reach a target position in a sequence \
          where each move advances by one of a small fixed set of offsets"
    BAD:  "determining the alphabetical order implied by an alien \
          language's word list"
    GOOD: "recovering a total order over a set of symbols from pairwise \
          precedence constraints"
- Do not name the specific problem or reuse phrasing copied from the problem \
  statement -- this includes synonyms and paraphrases of its entities, not \
  just its exact words.
- "mechanism" must not say things like "as described above" or otherwise \
  depend on the techniques list to make sense.
- Output ONLY the JSON object. No markdown fences, no leading or trailing text.
"""

# SYSTEM_PROMPT = """You analyze competitive-programming solutions to extract their \
# algorithmic structure: the individual techniques involved, and the mechanism \
# that ties them together.
 
# Given a problem's description and a working solution, respond with ONLY a \
# single JSON object (no markdown code fences, no commentary before or after) \
# in exactly this shape:
 
# {
#   "techniques": [
#     {
#       "name": "<short technique/data-structure name, e.g. \\"heap\\", \\"hash set\\", \\"prefix sum\\">",
#       "role": "<1 sentence, in plain language, on what this technique accomplishes in this solution. For niche or compound techniques, explain the underlying mechanism itself rather than just naming it -- e.g. for lazy deletion from a heap: \\"stale entries are left in place and skipped over lazily when they surface at the top, rather than being removed immediately on invalidation.\\">"
#     }
#   ],
#   "mechanism": "<2-4 sentences describing how the techniques above work together to solve the problem. Write this around the underlying goal/invariant (e.g. maintaining a running target sum, finding the nearest valid boundary) rather than leading with any single technique's name. This field must stand alone and read clearly without the techniques list for context -- it is what gets embedded for similarity search.>"
# }
 
# Rules:
# - List EVERY technique/data structure that meaningfully contributes to the \
#   solution, not just the most prominent one. Do not rank or single out a \
#   "primary" one -- if three techniques are all load-bearing, list three.
# - Describe the abstract structure of the problem, not its specific entities. \
#   E.g. say "a collection of items with a target sum" rather than "an array of \
#   coin denominations."
# - Do not name the specific problem or reuse phrasing copied from the problem \
#   statement where avoidable.
# - "mechanism" must not say things like "as described above" or otherwise \
#   depend on the techniques list to make sense.
# - Output ONLY the JSON object. No markdown fences, no leading or trailing text.
# """

# SYSTEM_PROMPT = """You analyze competitive-programming solutions to extract the \
# underlying algorithmic technique, abstracted away from the specific problem.

# Given a problem's description and a working solution, output a short, \
# technique-focused description in EXACTLY this template:

# <Primary technique / data structure>: <1-2 sentences on the core mechanism or \
# invariant that makes the solution work>.

# Rules:
# - Describe the abstract structure of the problem, not its specific entities. \
#   E.g. say "a collection of items with a target sum" rather than "an array of \
#   coin denominations."
# - Do not name the specific problem or reuse phrasing copied from the problem \
#   statement where avoidable.
# - Focus on the HOW (the mechanism/invariant), not the WHAT (the surface goal).
# - Keep it to 1-2 sentences after the technique label. No preamble, no markdown.
# """


USER_TEMPLATE = """Problem description:
{description}

Solution code:
{code}
"""


# Import dataset
import pandas as pd

splits = {'train': 'LeetCodeDataset-train.jsonl', 'test': 'LeetCodeDataset-test.jsonl'}
with pd.read_json("hf://datasets/newfacade/LeetCodeDataset/" + splits["train"], lines=True, chunksize=10) as reader:
    df = next(reader)
    df.rename(columns={'task_id': 'name', 'problem_description': 'description', 'completion': 'code'}, inplace=True)



def _user_content(problem: dict) -> str:
    return USER_TEMPLATE.format(
        description=problem["description"].strip(),
        code=problem["code"].strip(),
    )


def _parse_analysis(raw_text: str) -> dict:
    """
    Models don't always obey "no markdown fences" perfectly, so this pulls
    out the first {...} block rather than assuming raw_text is pure JSON.
    They also occasionally emit almost-valid JSON -- e.g. a stray backslash
    before the closing quote of the last string field, which makes the
    parser read it as an escaped quote and keep scanning for a real
    terminator that never comes ("Unterminated string"). Strict json.loads
    is tried first (cheap, handles the common case); only on failure do we
    pay for json_repair, which is built specifically for this class of
    almost-valid LLM JSON output. If neither works, the output is validated
    the same way either way, so a genuinely malformed/incomplete response
    still surfaces a clear error instead of silently returning garbage.
 
    pip install json-repair --break-system-packages
    """
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
 
    techniques = analysis.get("techniques")
    if not isinstance(techniques, list) or not techniques:
        raise ValueError(f"Missing/empty 'techniques' list: {analysis!r}")
    for t in techniques:
        if not isinstance(t, dict) or not t.get("name") or not t.get("role"):
            raise ValueError(f"Malformed technique entry: {t!r}")
 
    mechanism = analysis.get("mechanism")
    if not isinstance(mechanism, str) or not mechanism.strip():
        raise ValueError(f"Missing/empty 'mechanism': {analysis!r}")
 
    return analysis


class TruncatedResponseError(RuntimeError):
    """
    Raised when a response was cut off by the token limit before the model
    finished writing -- a token-budget problem, not a JSON-parsing problem.
    Distinguishing this from a genuinely malformed response matters because
    the fix is different (raise max_tokens / shrink the ask) and downstream
    symptoms (a dangling string fragment where a dict should be, an
    unterminated string, etc) can otherwise look like arbitrary parser bugs.
    """


def get_abstract(problem: dict) -> str:
    if PROVIDER == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=MODEL,
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_content(problem)}],
        )
        return msg.content[0].text.strip()

    if PROVIDER == "groq":
        from groq import Groq

        # max_retries=0: the default of 2 makes the SDK silently sleep out
        # the suggested Retry-After on a 429 before ever raising -- and our
        # daily-quota 429s carry multi-minute suggested waits, so that default
        # can burn minutes per call on an already-exhausted model before our
        # own model-fallback logic ever gets a chance to move on.
        client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=0)
        kwargs = dict(
            model=MODEL,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_content(problem)},
            ],
        )
        try:
            resp = client.chat.completions.create(reasoning_effort="low", **kwargs)
        except Exception as e:
            # Some models on Groq (groq/compound, groq/compound-mini, allam-2-7b)
            # reject reasoning_effort entirely rather than just ignoring it.
            if "reasoning_effort" in str(e):
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if choice.finish_reason == "length":
            reasoning = getattr(choice.message, "reasoning", None)
            raise TruncatedResponseError(
                f"Response truncated by max_tokens"
                f"(reasoning_tokens="
                f"{resp.usage.completion_tokens_details.reasoning_tokens if resp.usage.completion_tokens_details else '?'}"
                f", content chars so far={len(content)}). Raise "
                f"max_tokens, lower reasoning_effort, or shrink the "
                f"ask (fewer techniques). Partial content: "
                f"{content[:200]!r} / partial reasoning: "
                f"{str(reasoning)[:200]!r}"
            )
        return _parse_analysis(content)

    if PROVIDER == "ollama":
        import ollama

        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_content(problem)},
            ],
        )
        return resp["message"]["content"].strip()

    raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")


def check_embedding_similarity(abstracts: dict) -> None:
    """
    The thing that actually matters isn't whether these abstracts *read*
    similarly to a human -- it's whether their embeddings do. Short-text
    embeddings can be dominated by whatever's most salient early in the
    string (here: the technique label), so don't trust eyeballing the text;
    measure it. This checks the ordinal hypothesis: Two Sum and 3Sum should
    be closer to each other than either is to Climbing Stairs, even if the
    absolute similarity isn't sky-high.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
 
    print("\nLoading local embedding model (first run downloads ~80MB)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", token=os.environ.get("HF_TOKEN"))
 
    names = list(abstracts.keys())
    mechanisms = [abstracts[n]["mechanism"] for n in names]
    vectors = embed_model.encode(mechanisms, normalize_embeddings=True)
    # vectors = embed_model.encode(list(abstracts.values()), normalize_embeddings=True)
    cos_sim = np.dot(vectors, vectors.T)
 
    print("\nPairwise cosine similarity of the approach-abstract embeddings:")
    header = f"{'':22s}" + "".join(f"{n:>16s}" for n in names)
    print(header)
    for i, ni in enumerate(names):
        print(f"{ni:22s}" + "".join(f"{cos_sim[i, j]:16.3f}" for j in range(len(names))))

    return names, cos_sim
    


if __name__ == "__main__":
    abstracts = {}
    for idx, problem in df.iterrows():
        abstract = get_abstract(problem[['name', 'description', 'code']].to_dict())
        abstracts[problem["name"]] = abstract
        print(f"--- {problem['name']} ---")
        print(abstract)
        print()
 
    check_embedding_similarity(abstracts)