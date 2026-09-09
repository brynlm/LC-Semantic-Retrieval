import pickle

import categorical_similarity as cs
import motif_detectors as md

df = cs.load_dataset()
with open("official_editorial_code_cache.pkl", "rb") as f:
    official_code = pickle.load(f)

# positive cases: known monotonic-stack/deque solutions
positive_cases = ["sliding-window-maximum", "jump-game-vi"]

# negative cases: known NOT-monotonic-stack solutions, including one
# (course-schedule-ii) that's a deliberately tricky near-miss -- Kahn's
# algorithm also pops from and pushes to a container in a loop, but with a
# bare `while queue:` truthiness check, no comparison against the queue's own
# edge element.
negative_cases = ["two-sum", "climbing-stairs", "course-schedule-ii", "minimum-height-trees", "merge-intervals"]

print("=== POSITIVE CASES (should detect the motif) ===")
for name in positive_cases:
    code = official_code.get(name)
    if code is None:
        print(f"{name}: no official code cached, skipping")
        continue
    matches = md.detect_monotonic_stack(code)
    status = "DETECTED" if matches else "MISSED (false negative!)"
    print(f"{name}: {status} -- {matches}")

print("\n=== NEGATIVE CASES (should NOT detect the motif) ===")
for name in negative_cases:
    code = official_code.get(name)
    if code is None:
        row = df[df["name"] == name].iloc[0]
        code = row["code"]
    matches = md.detect_monotonic_stack(code)
    status = "correctly clean" if not matches else "FALSE POSITIVE!"
    print(f"{name}: {status} -- {matches}")
