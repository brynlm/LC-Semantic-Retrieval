import categorical_similarity as cs
import motif_detectors as md

df = cs.load_dataset()


def code_of(name):
    return df[df["name"] == name].iloc[0]["code"]


print("=== BINARY SEARCH: positive cases ===")
for name in ["binary-search", "search-insert-position", "sqrtx"]:
    matches = md.detect_binary_search(code_of(name))
    print(f"{name}: {'DETECTED' if matches else 'MISSED'} -- {matches}")

print("\n=== BINARY SEARCH: negative cases ===")
for name in ["two-sum", "climbing-stairs", "course-schedule-ii", "merge-intervals", "valid-palindrome", "container-with-most-water"]:
    matches = md.detect_binary_search(code_of(name))
    print(f"{name}: {'correctly clean' if not matches else 'FALSE POSITIVE'} -- {matches}")

print("\n=== TWO-POINTER CONVERGENCE: positive cases ===")
for name in ["valid-palindrome", "container-with-most-water", "3sum"]:
    matches = md.detect_two_pointer_convergence(code_of(name))
    print(f"{name}: {'DETECTED' if matches else 'MISSED'} -- {matches}")

print("\n=== TWO-POINTER CONVERGENCE: negative cases ===")
for name in ["two-sum", "climbing-stairs", "course-schedule-ii", "merge-intervals", "binary-search", "search-insert-position"]:
    matches = md.detect_two_pointer_convergence(code_of(name))
    print(f"{name}: {'correctly clean' if not matches else 'FALSE POSITIVE'} -- {matches}")
