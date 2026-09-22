import json

import categorical_similarity as cs
import test_approach_abstract as taa

taa.MODEL = "openai/gpt-oss-120b"

DESC_ONLY_USER_TEMPLATE = """Problem description:
{description}

No solution code is provided. Based on the description alone, describe the \
standard/canonical algorithmic approach a competitive programmer would use to \
solve this problem efficiently.
"""

TEST_NAMES = [
    "sliding-window-maximum",
    "longest-happy-prefix",
    "minimum-time-to-revert-word-to-initial-state-i",
    "minimum-time-to-revert-word-to-initial-state-ii",
]

df = cs.load_dataset()

for name in TEST_NAMES:
    row = df[df["name"] == name].iloc[0]
    user_content = DESC_ONLY_USER_TEMPLATE.format(description=row["description"].strip())

    from groq import Groq
    import os

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=taa.MODEL,
        max_tokens=1000,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": taa.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    content = resp.choices[0].message.content.strip()
    try:
        parsed = taa._parse_analysis(content)
        print(f"=== {name} (description-only) ===")
        print("techniques:", [t["name"] for t in parsed["techniques"]])
        print("mechanism:", parsed["mechanism"])
        print()
    except Exception as e:
        print(f"=== {name} FAILED TO PARSE: {e} ===")
        print(content)
        print()
