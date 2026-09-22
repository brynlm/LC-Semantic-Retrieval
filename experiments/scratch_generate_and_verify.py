"""
Test: generate our own solution code (independent of the dataset's sample),
verify it against the dataset's own test harness, and only if it passes, feed
that verified code into the normal (non-hybrid) abstract-generation prompt.

Restricted to the 4 diagnostic problems to keep token usage bounded.
"""
import os
import re
import signal
import traceback

from groq import Groq

import categorical_similarity as cs
import test_approach_abstract as taa

taa.MODEL = "openai/gpt-oss-120b"

CODEGEN_SYSTEM_PROMPT = """You are an expert competitive programmer. Given a \
problem description and a required method signature, write the single most \
standard, efficient, correct Python solution a strong competitive programmer \
would use. Respond with ONLY a Python code block containing the complete \
class definition matching the given signature -- no explanation, no markdown \
outside the code fence."""

CODEGEN_USER_TEMPLATE = """Problem description:
{description}

Required signature:
{starter_code}
"""

TEST_NAMES = [
    "sliding-window-maximum",
    "longest-happy-prefix",
    "minimum-time-to-revert-word-to-initial-state-i",
    "minimum-time-to-revert-word-to-initial-state-ii",
]

client = Groq(api_key=os.environ["GROQ_API_KEY"])
df = cs.load_dataset()


def extract_code(raw: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    return (m.group(1) if m else raw).strip()


class TimeoutError_(Exception):
    pass


def _handler(signum, frame):
    raise TimeoutError_()


def run_test(prompt_boilerplate: str, code: str, test_code: str, entry_point: str, timeout_s: int = 5) -> tuple[bool, str]:
    namespace = {}
    try:
        exec(prompt_boilerplate, namespace)
        exec(code, namespace)
        exec(test_code, namespace)
        candidate = eval(entry_point, namespace)
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_s)
        try:
            namespace["check"](candidate)
        finally:
            signal.alarm(0)
        return True, "PASSED"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


results = {}
for name in TEST_NAMES:
    row = df[df["name"] == name].iloc[0]
    user_content = CODEGEN_USER_TEMPLATE.format(
        description=row["description"].strip(), starter_code=row["starter_code"].strip()
    )
    resp = client.chat.completions.create(
        model=taa.MODEL,
        max_tokens=1500,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw_code = resp.choices[0].message.content.strip()
    code = extract_code(raw_code)

    passed, msg = run_test(row["prompt"], code, row["test"], row["entry_point"])
    print(f"=== {name}: {'PASS' if passed else 'FAIL'} ({msg}) ===")
    print(code)
    print()
    results[name] = {"code": code, "passed": passed, "msg": msg}

import pickle
with open("/tmp/generated_code_results.pkl", "wb") as f:
    pickle.dump(results, f)
