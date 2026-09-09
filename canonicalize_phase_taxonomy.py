"""
Canonicalizes the raw, freely-generated labels collected by
run_phase_taxonomy_discovery.py into a smaller set of canonical technique
names -- e.g. "dfs", "recursive dfs", "dfs with time tracking", and
"multithreaded dfs" should all collapse to one canonical "depth-first search"
label. Discovery found 35 distinct preprocessing labels (mostly fine as-is)
but 167 distinct algorithm labels across only 190 entries -- almost no
vocabulary reuse -- so this pass matters most for the algorithm phase.

Important scoping note: this is a *retrospective* merge of labels already
collected, not yet a closed vocabulary enforced on future classification
calls. Constraining test_phase_taxonomy.py's prompt to only emit canonical
labels going forward is a separate, later decision -- do that once the
algorithm vocabulary has seen enough breadth that a fixed list wouldn't
prematurely cut off legitimate new techniques.

This produces a raw_label -> canonical_label mapping per phase (one LLM call
per phase, since the two vocabularies are small enough to fit in one prompt
each) and applies it to the discovery cache so the canonicalized labels can
be inspected and, later, fed into tag_similarity_matrix-style comparison.

Usage:
    python canonicalize_phase_taxonomy.py
"""

import json
import os
import pickle
from collections import Counter, defaultdict

import test_phase_taxonomy as tpt

DISCOVERY_CACHE_PATH = os.path.join(os.path.dirname(__file__), "phase_taxonomy_discovery_cache.pkl")
MAPPING_CACHE_PATH = os.path.join(os.path.dirname(__file__), "phase_taxonomy_canonical_mapping.pkl")

PROVIDER = tpt.PROVIDER
MODEL = tpt.MODEL

SYSTEM_PROMPT = """You clean up a vocabulary of short algorithm/data-structure labels that \
were freely generated, one problem at a time, to describe the {phase_desc} \
step of many different solutions. Because they were generated independently \
per problem, the same underlying technique often ended up under several \
different strings -- sometimes with a problem-specific qualifier baked in \
(e.g. "dfs value assignment", "recursive dfs", "dfs with time tracking" are \
all just depth-first search).

You will be given a JSON array of [label, count] pairs, where count is how \
many times that exact label string appeared. Respond with ONLY a single JSON \
object (no markdown fences, no commentary) mapping EVERY given label \
(verbatim, exactly as given, used as the JSON key) to a single canonical \
label (the value):

{{
  "<raw label 1>": "<canonical label>",
  "<raw label 2>": "<canonical label>",
  ...
}}

Rules:
- Merge labels that name the same core technique even when worded \
  differently or padded with problem-specific detail -- e.g. "dfs (hierholzer's \
  algorithm)", "dfs island counting", and "recursive dfs" should all map to \
  the same canonical label if they're really just DFS (unless one names a \
  genuinely distinct, load-bearing algorithm on top of DFS, like Hierholzer's \
  algorithm specifically for Eulerian paths -- use judgment: a *named* \
  algorithm variant with its own identity is not just "dfs").
- Do NOT merge labels that are genuinely different techniques just because \
  they commonly co-occur (e.g. "binary search" and "sorting" stay separate).
- Prefer standard, widely-recognized algorithms/data-structure terminology \
  (the kind used in an algorithms textbook or competitive programming) for \
  the canonical label, even if no single raw label used that exact wording.
- The canonical label a group of raw labels maps to does not need to itself \
  appear among the raw labels.
- Every key in your output must be one of the given raw labels, copied \
  exactly (same case, same text). Every given raw label must appear as a key.
- Output ONLY the JSON object.
"""

USER_TEMPLATE = """Raw labels with occurrence counts:
{labels_json}
{known_canonicals_block}"""

KNOWN_CANONICALS_BLOCK = """
Canonical labels already established for OTHER raw labels in this same phase \
(from an earlier batch -- this list is split across several calls). Reuse \
one of these if it genuinely fits one of the raw labels below, rather than \
inventing a near-duplicate new one; only propose something new if nothing \
here fits:
{known_canonicals_json}
"""


def _build_prompt_input(labels: dict, known_canonicals: list[str]) -> str:
    pairs = sorted(labels.items(), key=lambda kv: -kv[1])
    known_block = (
        KNOWN_CANONICALS_BLOCK.format(known_canonicals_json=json.dumps(known_canonicals))
        if known_canonicals
        else ""
    )
    return USER_TEMPLATE.format(labels_json=json.dumps(pairs), known_canonicals_block=known_block)


def _call_llm(system_prompt: str, user_content: str, max_tokens: int = 4000) -> str:
    """Same provider branching as test_phase_taxonomy.get_phase_taxonomy, just
    parameterized on prompt/max_tokens instead of a fixed schema -- this call
    needs a much bigger max_tokens since it echoes back every raw label."""
    if PROVIDER == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return msg.content[0].text.strip()

    if PROVIDER == "groq":
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if choice.finish_reason == "length":
            raise tpt.TruncatedResponseError(
                f"Response truncated by max_tokens (content chars so far="
                f"{len(content)}). Raise max_tokens or split the label list "
                f"into smaller batches. Partial content: {content[:300]!r}"
            )
        return content

    if PROVIDER == "ollama":
        import ollama

        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return resp["message"]["content"].strip()

    raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")


def iter_canonicalize_batches(labels: Counter, phase_desc: str, batch_size: int, already_mapped: dict):
    """
    Yields one {raw_label: canonical_label} dict per batch of at most
    `batch_size` not-yet-mapped labels (largest-count-first, so the most
    influential labels get canonicalized -- and contribute to
    `known_canonicals` for later batches -- first). Splitting into small
    batches keeps each call's token footprint low, which matters when the
    provider's rate limit is nearly exhausted (see module docstring/git
    history: the full 167-label algorithm vocabulary in one call kept getting
    429'd against Groq's free-tier daily token cap). `already_mapped` seeds
    the "known canonical labels" context so batches stay consistent with each
    other despite being separate calls.
    """
    remaining = [(l, c) for l, c in labels.most_common() if l not in already_mapped]
    system_prompt = SYSTEM_PROMPT.format(phase_desc=phase_desc)

    for i in range(0, len(remaining), batch_size):
        chunk = dict(remaining[i : i + batch_size])
        known_canonicals = sorted(set(already_mapped.values()))
        user_content = _build_prompt_input(chunk, known_canonicals)
        raw_text = _call_llm(system_prompt, user_content, max_tokens=max(800, batch_size * 30))
        batch_mapping = _parse_mapping(raw_text, expected_keys=set(chunk))
        already_mapped.update(batch_mapping)
        yield batch_mapping


def _parse_mapping(raw_text: str, expected_keys: set) -> dict:
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
        mapping = json.loads(candidate)
    except json.JSONDecodeError as strict_err:
        try:
            import json_repair

            mapping = json_repair.loads(candidate)
        except Exception:
            raise ValueError(
                f"Model output wasn't valid JSON ({strict_err}), and "
                f"json_repair couldn't recover it either: {raw_text!r}"
            ) from strict_err
        if not isinstance(mapping, dict):
            raise ValueError(f"json_repair recovered non-dict output ({type(mapping)}): {raw_text!r}")
        print("  [warning: JSON needed repair]")

    missing = expected_keys - set(mapping)
    if missing:
        print(f"  [warning: {len(missing)} raw labels missing from mapping, left unmapped (identity): {missing}")
        for label in missing:
            mapping[label] = label

    return {k: v for k, v in mapping.items() if k in expected_keys}


def apply_mapping(cache: dict, mapping_by_phase: dict) -> dict:
    """Remaps every {label, role} entry in `cache` (as produced by
    run_phase_taxonomy_discovery.py) with a "canonical_label" field, leaving
    the original "label"/"role" untouched for traceability."""
    canonicalized = {}
    for name, taxonomy in cache.items():
        new_taxonomy = {}
        for phase, entries in taxonomy.items():
            mapping = mapping_by_phase.get(phase, {})
            new_taxonomy[phase] = [
                {**entry, "canonical_label": mapping.get(entry["label"].strip().lower(), entry["label"])}
                for entry in entries
            ]
        canonicalized[name] = new_taxonomy
    return canonicalized


PHASE_DESCRIPTIONS = {
    "preprocessing": "PREPROCESSING (reorganizing/transforming the input before the core logic runs)",
    "algorithm": "ALGORITHM (the technique that derives the answer)",
}
# preprocessing's 35-label vocabulary fit in one call; algorithm's 167-label
# vocabulary did not (see iter_canonicalize_batches docstring) -- batch it.
BATCH_SIZE_BY_PHASE = {"preprocessing": 200, "algorithm": 25}


def run() -> tuple[dict, dict]:
    with open(DISCOVERY_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    # Resumable at the batch level, not just the phase level -- a 167-label
    # vocabulary needs several calls, and a rate limit can hit partway
    # through; save after every batch, not only once a whole phase finishes.
    mapping_by_phase = {}
    if os.path.exists(MAPPING_CACHE_PATH):
        with open(MAPPING_CACHE_PATH, "rb") as f:
            mapping_by_phase = pickle.load(f)

    for phase, phase_desc in PHASE_DESCRIPTIONS.items():
        labels = Counter(e["label"].strip().lower() for t in cache.values() for e in t.get(phase, []))
        phase_mapping = mapping_by_phase.setdefault(phase, {})
        n_remaining = len(set(labels) - set(phase_mapping))
        if n_remaining == 0:
            print(f"Skipping {phase} (already fully canonicalized, cached)")
            continue

        print(f"Canonicalizing {phase}: {len(labels)} distinct raw labels, {n_remaining} remaining...")
        batch_size = BATCH_SIZE_BY_PHASE.get(phase, 200)
        for batch_mapping in iter_canonicalize_batches(labels, phase_desc, batch_size, phase_mapping):
            with open(MAPPING_CACHE_PATH, "wb") as f:
                pickle.dump(mapping_by_phase, f)
            print(f"  ...{len(phase_mapping)}/{len(labels)} mapped so far")

    canonicalized_cache = apply_mapping(cache, mapping_by_phase)
    return mapping_by_phase, canonicalized_cache


def summarize(mapping_by_phase: dict) -> None:
    for phase, mapping in mapping_by_phase.items():
        groups = defaultdict(list)
        for raw, canonical in mapping.items():
            groups[canonical].append(raw)
        print(f"\n=== {phase}: {len(mapping)} raw labels -> {len(groups)} canonical labels ===")
        for canonical, raws in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"{len(raws):3d}  {canonical}")
            if len(raws) > 1:
                for r in raws:
                    print(f"       <- {r}")


if __name__ == "__main__":
    mapping_by_phase, canonicalized_cache = run()
    with open(os.path.join(os.path.dirname(__file__), "phase_taxonomy_canonicalized_cache.pkl"), "wb") as f:
        pickle.dump(canonicalized_cache, f)
    summarize(mapping_by_phase)
