import os
import re
import signal
import pickle

from dotenv import load_dotenv
load_dotenv()
from groq import Groq

import categorical_similarity as cs
import test_approach_abstract as taa

taa.MODEL = "openai/gpt-oss-120b"
MODEL = taa.MODEL

CODEGEN_SYSTEM_PROMPT = """You are an expert competitive programmer. Given a \
problem description and a required method signature, write the single most \
standard, efficient, correct Python solution a strong competitive programmer \
would use. Respond with ONLY a Python code block containing the complete \
class definition matching the given signature -- no explanation, no markdown \
outside the code fence. Handle all edge cases carefully, including boundary \
indices."""

# 1 still-unresolved regression + 5 diverse "improved" pairs
PAIRS = [
    ("describe-the-painting", "shifting-letters-ii"),
    ("minimum-height-trees", "course-schedule-ii"),
    ("capacity-to-ship-packages-within-d-days", "split-array-largest-sum"),
    ("number-of-provinces", "number-of-connected-components-in-an-undirected-graph"),
    ("reverse-pairs", "count-of-smaller-numbers-after-self"),
    ("different-ways-to-add-parentheses", "the-score-of-students-solving-math-expression"),
]
NAMES = sorted({n for pair in PAIRS for n in pair})
print(f"{len(NAMES)} unique names: {NAMES}")

client = Groq(api_key=os.environ["GROQ_API_KEY"])
df = cs.load_dataset()


def extract_code(raw: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    return (m.group(1) if m else raw).strip()


def _handler(signum, frame):
    raise TimeoutError()


def run_test(prompt_boilerplate, code, test_code, entry_point, timeout_s=5):
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
        return False, f"{type(e).__name__}: {str(e)[:150]}"


token_log = []  # (name, call_type, attempt, prompt_tokens, completion_tokens, total_tokens)
verified_code = {}

for name in NAMES:
    row = df[df["name"] == name].iloc[0]
    user_content = f"Problem description:\n{row['description'].strip()}\n\nRequired signature:\n{row['starter_code'].strip()}\n"

    passed = False
    for attempt in range(1, 4):
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=2000, reasoning_effort="low", temperature=0.7 if attempt > 1 else 0.0,
            messages=[{"role": "system", "content": CODEGEN_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
        )
        u = resp.usage
        token_log.append((name, "codegen", attempt, u.prompt_tokens, u.completion_tokens, u.total_tokens))
        raw_code = (resp.choices[0].message.content or "").strip()
        code = extract_code(raw_code)
        passed, msg = run_test(row["prompt"], code, row["test"], row["entry_point"])
        print(f"  [{name}] codegen attempt {attempt}: {'PASS' if passed else 'FAIL: ' + msg}")
        if passed:
            verified_code[name] = code
            break
    if not passed:
        print(f"  [{name}] *** never passed after 3 attempts, using dataset's original code as fallback ***")
        verified_code[name] = row["code"]

with open("/tmp/broader_verified_code.pkl", "wb") as f:
    pickle.dump(verified_code, f)
with open("/tmp/broader_token_log.pkl", "wb") as f:
    pickle.dump(token_log, f)

# now generate abstracts from verified code
abstract_results = {}
for name in NAMES:
    row = df[df["name"] == name].iloc[0]
    problem = {"name": name, "description": row["description"], "code": verified_code[name]}
    from groq import Groq as _Groq
    resp = client.chat.completions.create(
        model=MODEL, max_tokens=1000, reasoning_effort="low",
        messages=[{"role": "system", "content": taa.SYSTEM_PROMPT}, {"role": "user", "content": taa._user_content(problem)}],
    )
    u = resp.usage
    token_log.append((name, "abstract", 1, u.prompt_tokens, u.completion_tokens, u.total_tokens))
    content = (resp.choices[0].message.content or "").strip()
    abstract_results[name] = taa._parse_analysis(content)
    print(f"  [{name}] abstract generated")

with open("/tmp/broader_abstracts.pkl", "wb") as f:
    pickle.dump(abstract_results, f)
with open("/tmp/broader_token_log.pkl", "wb") as f:
    pickle.dump(token_log, f)

total_tokens = sum(row[5] for row in token_log)
print(f"\nTotal tokens used: {total_tokens} across {len(NAMES)} problems ({total_tokens/len(NAMES):.0f}/problem avg)")
print(f"Total API calls: {len(token_log)} ({len(token_log)/len(NAMES):.1f}/problem avg)")
