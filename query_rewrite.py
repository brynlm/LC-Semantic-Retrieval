"""
Turns a user's free-text description of what they're looking for into a
synthetic mechanism-abstract paragraph, in the SAME register the corpus's own
abstracts were generated in (see test_approach_abstract.py's SYSTEM_PROMPT),
so it can be embedded with the same model and land in a comparable region of
embedding space -- a raw user query ("something with two pointers closing in
from both ends") is written in a different register than "the two-pointer
scan converges toward the middle while comparing endpoint sums", and MiniLM's
embedding space is sensitive to that register gap, not just topical overlap.
This is the one genuinely new LLM-in-the-loop feature this project asked
for: query rewriting/expansion at retrieval time, not just corpus generation.

Deliberately NOT reusing scratch_code_only_abstracts.py's retry-with-backoff
logic -- that's tuned for a long unattended batch job where waiting out a
short rate-limit is free. This runs inside a live HTTP request, so it tries
each model in the chain once, moves on immediately on any failure, and never
sleeps -- a slow but complete answer beats a fast timeout, but a HANGING
request is worse than a quick failure the caller can retry.

Usage:
    import query_rewrite as qr
    mechanism_text = qr.rewrite_query("something with two pointers closing in from both ends")
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_CHAIN = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-safeguard-20b"]

SYSTEM_PROMPT = """A user is describing, in their own words, the kind of algorithmic \
problem or solution mechanism they want to find. Rewrite their description as a single \
paragraph (2-4 sentences) in the exact register used by a corpus of solution-mechanism \
abstracts, so it can be embedded and compared against them directly.

Rules (identical to the corpus's own generation rules, since this text must land in the \
same region of embedding space):
- Use ONLY bare computer-science vocabulary: "graph", "node", "directed edge", "interval", \
  "array", "sequence", "tree", "set", "state", "connected component", "prefix sum", "sliding \
  window", "monotonic stack", and so on. If the user's own words use a domain-flavor noun \
  (e.g. "course", "student", "stone", "building"), translate it to the underlying CS object \
  rather than repeating it.
- Describe the underlying goal/invariant and how the technique(s) achieve it, the way a \
  solution's own mechanism would be described -- not a restatement or summary of the user's \
  request.
- If the user's description is vague or high-level (e.g. "graph problems" or "something \
  with dynamic programming"), write the most representative, concrete mechanism paragraph \
  for that request rather than staying abstract or hedging.
- If the user names a specific technique (e.g. "two pointers", "topological sort", "binary \
  search on the answer"), center the paragraph on that technique's actual mechanism.
- Output ONLY the rewritten paragraph. No preamble, no quotes, no markdown, no commentary.
"""


def rewrite_query(prompt: str) -> str | None:
    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=0)
    for model in MODEL_CHAIN:
        try:
            kwargs = dict(
                model=model,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            # gpt-oss models are reasoning models that otherwise burn a large,
            # variable chunk of max_tokens on invisible reasoning tokens
            # before writing anything visible (observed: 348/600 tokens spent
            # reasoning on one test prompt, leaving too little room for the
            # actual paragraph and truncating it mid-sentence) -- same fix as
            # scratch_code_only_abstracts.py's get_code_only_abstract, with
            # the same fallback since non-reasoning models (qwen, safeguard)
            # reject the parameter outright.
            try:
                resp = client.chat.completions.create(reasoning_effort="low", **kwargs)
            except Exception as e:
                if "reasoning_effort" in str(e):
                    resp = client.chat.completions.create(**kwargs)
                else:
                    raise
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:
            continue
    return None
