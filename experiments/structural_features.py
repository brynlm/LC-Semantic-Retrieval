"""
General-purpose structural feature extraction (phase 1: loop shape,
branching, recursion, recurrence-lookback).

Unlike motif_detectors.py -- which asks "is this one specific named pattern
present, yes/no" -- this extracts a dense feature vector per problem: loop
counts/nesting, branch/early-exit presence, recursion, memoization, and
recurrence lookback depth. The payoff is generality: a single numeric feature
like recurrence_lookback clusters climbing-stairs/fibonacci-number (lookback
2) with n-th-tribonacci-number (lookback 3) automatically, without ever
writing a "Fibonacci-like" detector by hand.

This is still pure AST-shape matching, not real data-flow analysis, though
phase 2 (loop_carried_state, mutable_structures, monotonic_vars,
index_relationship) does a lightweight def-use pass: a name counts as
loop-carried if it's initialized before a loop and then touched inside it,
where "touched" includes mutation (`.append()`, `d[x]=i`, `heapq.heappush`),
not just reassignment. recurrence_lookback recognizes three coding idioms for
the same linear recurrence (explicit array, sequential rolling scalars,
single-statement tuple rotation) and correctly assigns climbing-stairs and
fibonacci-number lookback 2, n-th-tribonacci-number lookback 3, despite each
using a different one of the three idioms.

Known gaps: monotonic_vars only recognizes self-referential arithmetic
(`x += 1`, `x = x - 1`); it does NOT catch a bound reassigned to a *derived*
variable like binary search's `l = mid + 1` / `r = mid`, so binary search's
own bounds don't show up as monotonic even though they are. running-max/min
patterns (`best = max(best, x)`) also aren't recognized as monotonicity
evidence. Both are known, documented limitations, not silent failures.

Comprehensions/generator expressions (`ListComp`, `SetComp`, `DictComp`,
`GeneratorExp`) are recognized as loop-like structure: each contributes to
num_loops/nesting/for_loop_styles (tagged "comprehension") the same way a
For/While does, and each generator clause's `.ifs` filter conditions count as
branch-inside-loop evidence. This fixes problems like vowels-of-all-substrings
(`sum((i+1)*(n-i) for i,c in enumerate(word) if c in 'aeiou')`) or
stone-game-vi (a list comprehension), which previously showed all-zero
features despite genuine per-element computation. Two deliberate
simplifications: a single comprehension node with multiple `for` clauses
(`[x for x in a for y in b]`) is NOT treated as two nested levels (rare in
practice, and comprehensions can't hold Assign/AugAssign statements anyway, so
phase-2 loop-carried-state/monotonic detection naturally contributes nothing
for them -- correct, since a comprehension has no internal accumulator of its
own). Closed-form/O(1) solutions with no iteration at all (`unique-paths`'
factorial formula, `valid-boomerang`'s single cross-product check) still show
all-zero features -- this is a permanent scope limit, not a bug, since there
is genuinely no loop-like structure to find.

Usage:
    import structural_features as sf
    features = sf.extract_features(code_string)
"""

import ast
from dataclasses import dataclass, field

from motif_detectors import _var_deltas_in_stmt as _md_var_deltas


@dataclass
class StructuralFeatures:
    num_loops: int = 0
    max_loop_nesting_depth: int = 0
    loop_nesting_pairs: set = field(default_factory=set)  # e.g. {"for>while"}
    for_loop_styles: set = field(default_factory=set)  # e.g. {"range_index", "direct_iteration"}
    branch_inside_loop: bool = False
    num_branches_in_loop: int = 0
    early_exit_in_loop: bool = False
    recursion: bool = False
    memoization: bool = False
    recurrence_lookback: int | None = None
    recurrence_combinator: str | None = None  # "sum"/"max"/"min"/"or"/"and"
    recursion_style: str | None = None  # "combined"/"backtrack"/"traversal"
    has_repairs: bool = False
    has_grows: bool = False
    loop_carried_state: set = field(default_factory=set)
    mutable_structures: set = field(default_factory=set)
    monotonic_vars: dict = field(default_factory=dict)  # {name: "increasing"/"decreasing"}
    index_relationship: set = field(default_factory=set)  # {"parallel", "converging"}

    @property
    def recurrence(self) -> bool:
        return self.recurrence_lookback is not None

    def as_dict(self) -> dict:
        d = {
            "num_loops": self.num_loops,
            "max_loop_nesting_depth": self.max_loop_nesting_depth,
            "loop_nesting_pairs": sorted(self.loop_nesting_pairs),
            "for_loop_styles": sorted(self.for_loop_styles),
            "branch_inside_loop": self.branch_inside_loop,
            "num_branches_in_loop": self.num_branches_in_loop,
            "early_exit_in_loop": self.early_exit_in_loop,
            "recursion": self.recursion,
            "memoization": self.memoization,
            "recurrence": self.recurrence,
            "recurrence_lookback": self.recurrence_lookback,
            "recurrence_combinator": self.recurrence_combinator,
            "recursion_style": self.recursion_style,
            "has_repairs": self.has_repairs,
            "has_grows": self.has_grows,
            "loop_carried_state": sorted(self.loop_carried_state),
            "mutable_structures": sorted(self.mutable_structures),
            "monotonic_vars": dict(sorted(self.monotonic_vars.items())),
            "num_monotonic_vars": len(self.monotonic_vars),
            "index_relationship": sorted(self.index_relationship),
        }
        return d


def _name_of(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


_COMPREHENSION_TYPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _loop_type(node: ast.AST) -> str | None:
    if isinstance(node, ast.For):
        return "for"
    if isinstance(node, ast.While):
        return "while"
    if isinstance(node, _COMPREHENSION_TYPES):
        return "comprehension"
    return None


def _walk_loops(node: ast.AST, depth: int = 0, parent_type: str | None = None):
    """Yields (loop_node, depth, parent_loop_type) for every For/While/
    comprehension (ListComp/SetComp/DictComp/GeneratorExp) reachable from
    `node`, honoring nesting depth through intervening non-loop constructs
    (an if/try between two loops still counts as nesting, and a comprehension
    buried inside a Call like `sum(x for x in ...)` is still found) and
    descending into nested function defs (helper functions like a recursive
    `dfs` defined inside the main method are part of the same algorithm, not
    a separate scope worth ignoring)."""
    for child in ast.iter_child_nodes(node):
        loop_type = _loop_type(child)
        if loop_type:
            yield child, depth, parent_type
            yield from _walk_loops(child, depth + 1, loop_type)
        else:
            yield from _walk_loops(child, depth, parent_type)


def _for_loop_style(node: ast.For) -> str:
    it = node.iter
    if isinstance(it, ast.Call):
        func_name = _name_of(it.func) or (it.func.attr if isinstance(it.func, ast.Attribute) else None)
        if func_name == "range":
            return "range_index"
        if func_name == "enumerate":
            return "enumerate"
        if func_name == "zip":
            return "zip"
        if func_name == "reversed":
            return "reversed"
        return "other_call"
    if isinstance(it, (ast.Name, ast.Attribute, ast.Subscript)):
        return "direct_iteration"
    return "other"


def _contains(node: ast.AST, types: tuple) -> bool:
    return any(isinstance(n, types) for n in ast.walk(node))


def _count_distinct(nodes_iter) -> int:
    return len({id(n) for n in nodes_iter})


def _function_defs(tree: ast.AST) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _calls_in(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                yield n.func.id
            elif isinstance(n.func, ast.Attribute):
                yield n.func.attr


def _detect_recursion(tree: ast.AST) -> bool:
    """True if any function (including a nested helper, e.g. a `dfs` defined
    inside the main method) calls itself by name -- either a bare call
    (`dfs(...)`) or a method call (`self.dfs(...)`)."""
    for fn in _function_defs(tree):
        if fn.name in set(_calls_in(fn)):
            return True
    return False


_CACHE_NAME_HINTS = {"memo", "cache", "dp", "seen", "visited"}


def _looks_like_cache_init(value: ast.AST) -> bool:
    """True if `value` initializes something shaped like a memo table: an
    empty dict/set (`{}`, `dict()`, `set()`, `defaultdict(...)`), or a
    fill-value list (`[-1] * n`, `[0] * n`)."""
    if isinstance(value, ast.Dict) and not value.keys:
        return True
    if isinstance(value, ast.Set) and not value.elts:
        return True
    if isinstance(value, ast.Call):
        fname = _name_of(value.func) or (value.func.attr if isinstance(value.func, ast.Attribute) else None)
        if fname in ("dict", "set", "defaultdict"):
            return True
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
        return isinstance(value.left, ast.List) or isinstance(value.right, ast.List)
    return False


_CACHE_DECORATOR_NAMES = {"cache", "lru_cache"}


def _decorator_name(node: ast.AST) -> str | None:
    """Name of a decorator, handling both bare (`@cache`) and called
    (`@lru_cache(maxsize=None)`) forms, and both bare-name and
    `functools.cache`-style attribute access."""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _has_cache_decorator(fn) -> bool:
    return any(_decorator_name(d) in _CACHE_DECORATOR_NAMES for d in fn.decorator_list)


def _detect_memoization(tree: ast.AST) -> bool:
    """Heuristic: some name is initialized as a cache-shaped container
    (dict/set/fill-value list) AND that same name is later either
    subscripted/checked-for-membership as a guard, or written into, within a
    function that also recurses. This is intentionally approximate -- it
    will miss e.g. a memo table implemented as a plain closure variable
    captured by reference with no recognizable init shape.

    Also catches the `@cache`/`@lru_cache` decorator idiom directly (very
    common in this dataset's canonical solutions) -- a decorated recursive
    function IS memoized regardless of whether it has any hand-written
    cache-shaped container at all, so this check runs independently of the
    manual-cache-container heuristic below, not as a fallback for it."""
    if any(_has_cache_decorator(fn) and fn.name in set(_calls_in(fn)) for fn in _function_defs(tree)):
        return True
    if not _detect_recursion(tree):
        return False
    cache_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = _name_of(node.targets[0])
            if name and (_looks_like_cache_init(node.value) or any(h in name.lower() for h in _CACHE_NAME_HINTS)):
                cache_names.add(name)
    if not cache_names:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _name_of(node.value) in cache_names:
            return True
        if isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            sides = [node.left, *node.comparators]
            if any(_name_of(s) in cache_names for s in sides):
                return True
    return False


def _term_count(node: ast.AST, is_term) -> int:
    return sum(1 for n in ast.walk(node) if is_term(n))


def _combinator_of(node: ast.AST, is_term) -> str | None:
    """Classifies how the "terms" of a recurrence (per the `is_term`
    predicate -- e.g. "is a recursive self-call") are combined within `node`:
    MAX/MIN if at least one term sits inside a max()/min() call (covers both
    `max(f(a), f(b))` and an accumulator pattern like `ans = max(ans,
    f(x))`), SUM if 2+ terms are joined by `+`, OR/AND if 2+ are joined by a
    boolean or/and. Checked in that priority order since max/min wrapping a
    single term alongside an unrelated value is still a real combinator
    (the recurrence chooses between this branch and an alternative), while
    +/or/and need 2+ terms to mean anything (an accumulator pattern for
    those doesn't exist the way it does for max/min)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fname = _name_of(n.func) or (n.func.attr if isinstance(n.func, ast.Attribute) else None)
            if fname in ("max", "min") and _term_count(n, is_term) >= 1:
                return fname
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) and _term_count(n, is_term) >= 2:
            return "sum"
    for n in ast.walk(node):
        if isinstance(n, ast.BoolOp) and _term_count(n, is_term) >= 2:
            return "or" if isinstance(n.op, ast.Or) else "and"
    return None


def _is_self_call(node: ast.AST, fn_name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == fn_name
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == fn_name
    return False


def _recursive_combinator(fn) -> str | None:
    """Combinator for a recursive function's own recurrence: how its
    self-calls' return values get combined. Distinguishes, e.g., house-robber
    (`max(nums[i] + dfs(i + 2), dfs(i + 1))` -> "max", an optimization
    recurrence) from a same-lookback-shape counting recurrence combined via
    "sum" -- the two are otherwise easy to conflate if only lookback distance
    is compared (see climbing-stairs vs house-robber, both lookback 2)."""
    if fn.name not in set(_calls_in(fn)):
        return None
    return _combinator_of(fn, lambda n: _is_self_call(n, fn.name))


def _is_backward_offset_subscript(node: ast.AST, array_name: str, loop_var: str) -> bool:
    if not (isinstance(node, ast.Subscript) and _name_of(node.value) == array_name):
        return False
    idx = node.slice
    return (isinstance(idx, ast.BinOp) and isinstance(idx.op, ast.Sub)
            and _name_of(idx.left) == loop_var and isinstance(idx.right, ast.Constant)
            and isinstance(idx.right.value, int))


def _recurrence_lookback_in_loop(loop: ast.For) -> tuple[int, str | None] | None:
    """If `loop`'s body contains `TARGET[i] = f(...)` where f references
    `TARGET[i - k]` for one or more constants k (the same array, offset
    backward from the same loop variable), returns (largest such k,
    combinator) -- combinator classified the same way as the recursive case
    (max/min/sum/or/and over the backward-offset terms, via
    _combinator_of), so e.g. edit-distance's `min(f[i-1][j], f[i][j-1],
    f[i-1][j-1]) + 1` tags "min" and n-th-tribonacci-number's editorial-style
    `dp[i-1]+dp[i-2]+dp[i-3]` tags "sum", the same as their
    rolling-scalar/recursive equivalents. Only recognizes the explicit-array
    form, not rolling-variable equivalents (`a, b = b, a + b`) -- see module
    docstring."""
    if not (isinstance(loop.target, ast.Name)):
        return None
    loop_var = loop.target.id
    best_lookback = None
    best_combinator = None
    for stmt in ast.walk(loop):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Subscript) and _name_of(target.slice) == loop_var):
            continue
        array_name = _name_of(target.value)
        if not array_name:
            continue
        found_here = False
        for node in ast.walk(stmt.value):
            if _is_backward_offset_subscript(node, array_name, loop_var):
                k = node.slice.right.value
                if best_lookback is None or k > best_lookback:
                    best_lookback = k
                found_here = True
        if found_here:
            combinator = _combinator_of(stmt.value, lambda n: _is_backward_offset_subscript(n, array_name, loop_var))
            if combinator is not None:
                best_combinator = combinator
    if best_lookback is None:
        return None
    return best_lookback, best_combinator


def _sum_of_names(value: ast.AST) -> list[str] | None:
    """If `value` is a bare Name, or a chain of `Add` over bare Names
    (`a + b`, `a + b + c`, ...), returns the flattened list of operand names;
    else None."""
    if isinstance(value, ast.Name):
        return [value.id]
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        left, right = _sum_of_names(value.left), _sum_of_names(value.right)
        if left and right:
            return left + right
    return None


def _simple_assign(stmt: ast.stmt) -> tuple[str, ast.AST] | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        name = _name_of(stmt.targets[0])
        if name:
            return name, stmt.value
    return None


def _rolling_recurrence_lookback(body: list[ast.stmt]) -> int | None:
    """Recognizes the rolling-scalar-variable idiom for a linear recurrence,
    e.g. climbing-stairs' `third = first + second; first = second; second =
    third` -- a sum of K bare-name operands, immediately followed by K
    sequential assignments that rotate those K variables forward one
    position, with the freshly computed sum taking the last slot. Returns K
    (the number of terms summed = how many prior states the recurrence
    depends on) for the first such block found, or None."""
    for i, stmt in enumerate(body):
        assign = _simple_assign(stmt)
        if not assign:
            continue
        target_name, value = assign
        operand_names = _sum_of_names(value)
        if not operand_names or len(operand_names) < 2 or isinstance(value, ast.Name):
            continue
        k = len(operand_names)
        rotate_chain = operand_names[1:] + [target_name]
        ok = True
        for offset, expected_value_name in enumerate(rotate_chain):
            idx = i + 1 + offset
            if idx >= len(body):
                ok = False
                break
            next_assign = _simple_assign(body[idx])
            if not next_assign:
                ok = False
                break
            tgt, val = next_assign
            if tgt != operand_names[offset] or _name_of(val) != expected_value_name:
                ok = False
                break
        if ok:
            return k
    return None


def _tuple_rolling_recurrence_lookback(stmt: ast.stmt) -> int | None:
    """Recognizes the single-statement tuple-rotation idiom, e.g.
    fibonacci-number's `a, b = b, a + b`: target is a K-tuple of names
    (t1..tK), value is a K-tuple where the first K-1 elements are a pure
    left-shift (v1=t2, v2=t3, ...) and the last element sums exactly the K
    target names. Returns K if matched, else None."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
        return None
    target, value = stmt.targets[0], stmt.value
    if not (isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)):
        return None
    if len(target.elts) != len(value.elts) or len(target.elts) < 2:
        return None
    target_names = [_name_of(e) for e in target.elts]
    if not all(target_names):
        return None
    k = len(target_names)
    for i in range(k - 1):
        if _name_of(value.elts[i]) != target_names[i + 1]:
            return None
    last_operands = _sum_of_names(value.elts[-1])
    if not last_operands or isinstance(value.elts[-1], ast.Name):
        return None
    if set(last_operands) != set(target_names):
        return None
    return k


def _assign_pairs(stmt: ast.stmt) -> list[tuple[str, ast.AST]]:
    """Returns [(target_name, value_expr), ...] for a statement -- handling
    simple assignment, tuple/list-unpacking assignment (each element paired
    with its corresponding value), and AugAssign (paired with the bare
    target, since there's no separate "value expression" to classify a role
    from)."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value = stmt.targets[0], stmt.value
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts):
            return [(_name_of(t), v) for t, v in zip(target.elts, value.elts) if _name_of(t)]
        name = _name_of(target)
        return [(name, value)] if name else []
    if isinstance(stmt, ast.AugAssign):
        name = _name_of(stmt.target)
        return [(name, stmt.target)] if name else []
    return []


_MUTATION_METHODS = {
    "append", "appendleft", "pop", "popleft", "add", "remove", "discard",
    "update", "extend", "insert", "clear", "setdefault",
}


def _touched_names_in(node: ast.AST) -> set[str]:
    """Every name whose underlying object is written to or mutated anywhere
    within `node`: direct (re)assignment, subscript-assignment (`d[x] = i`),
    or a mutating method call (`.append(...)`, `.add(...)`, `.pop()`, a bare
    `heapq.heappush(heap, x)` where the container is the first argument,
    ...). This is deliberately broader than "assigned" -- a mutable
    container (list/dict/set/deque/heap) essentially never gets literally
    reassigned inside a loop, it gets mutated in place, which a plain
    Assign/AugAssign scan would completely miss."""
    names = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            names.update(name for name, _ in _assign_pairs(n))
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Subscript):
                    base = _name_of(t.value)
                    if base:
                        names.add(base)
        elif isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and n.func.attr in _MUTATION_METHODS:
                base = _name_of(n.func.value)
                if base:
                    names.add(base)
            elif isinstance(n.func, ast.Attribute) and n.func.attr in ("heappush", "heappop", "heapify") and n.args:
                base = _name_of(n.args[0])
                if base:
                    names.add(base)
            elif isinstance(n.func, ast.Name) and n.func.id in ("heappush", "heappop", "heapify") and n.args:
                base = _name_of(n.args[0])
                if base:
                    names.add(base)
    return names


_GROW_METHODS = {"append", "appendleft", "add", "extend", "insert", "update"}
_SHRINK_METHODS = {"pop", "popleft", "remove", "discard"}


def _mutation_base(node: ast.AST, methods: set) -> str | None:
    """If `node` is a call to one of `methods` (as an attribute method, e.g.
    `stk.pop()`, or a bare heapq-style call, e.g. `heappush(h, x)`), returns
    the name of the structure being mutated; else None."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Attribute) and node.func.attr in methods:
        return _name_of(node.func.value)
    bare_equivalent = {"append": "heappush", "pop": "heappop"}
    for method, bare in bare_equivalent.items():
        if method in methods and isinstance(node.func, ast.Name) and node.func.id == bare and node.args:
            return _name_of(node.args[0])
    return None


def _detect_repairs(tree: ast.AST) -> bool:
    """REPAIRS relation: an inner `while` loop whose test references some
    structure, and whose body shrinks (pops/removes from) that SAME
    structure -- the "repeat until this invariant is restored" idiom shared
    by sliding-window shrink loops (`while c in ss: ss.remove(...)`),
    monotonic-stack pop loops (`while stk and ...: stk.pop()`), and bounded
    heaps (`while len(h) > 1: ... heappop(h) ...`), despite these three
    looking like unrelated named motifs at the AST-shape level. Requiring
    the SAME name in both the guard and the shrink keeps this from firing on
    an unrelated while loop that merely happens to contain some pop call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        guard_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        for stmt in node.body:
            for call in ast.walk(stmt):
                base = _mutation_base(call, _SHRINK_METHODS)
                if base and base in guard_names:
                    return True
    return False


def _detect_grows(tree: ast.AST) -> bool:
    """GROWS relation: the natural counterpart to REPAIRS -- a guard that
    tests a variable/subscript that was JUST mutated (via `+=`/`-=`)
    EARLIER IN THE SAME BLOCK, whose body reacts by growing some structure
    (the "relaxation" idiom: decrement an in-degree/remaining-count, and
    when it crosses a threshold, enqueue -- course-schedule's `indeg[j] -=
    1; if indeg[j] == 0: q.append(j)`).

    Deliberately restricted to a PRIOR SIBLING statement in the same block
    (not "anywhere in the enclosing loop", tried first and found too loose):
    a classic two-pointer merge step (`if A[i] < B[j]: res.append(A[i]); i
    += 1 else: res.append(B[j]); j += 1`, from basic-calculator-iv) also has
    an AugAssign of a tested name somewhere in the same loop, but with the
    OPPOSITE causality -- the mutation there is a CONSEQUENCE of the
    if/else branch choice, appearing textually after it, not a cause the
    guard is reacting to. Requiring same-block + prior-sibling order keeps
    the relaxation semantics (test what just changed) without matching the
    much more common "test, then do one of two mutations" shape."""
    for block in _all_bodies(tree):
        mutated_so_far = set()
        for stmt in block:
            if isinstance(stmt, ast.If):
                test_bases = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
                if test_bases & mutated_so_far:
                    for body_stmt in stmt.body:
                        for call in ast.walk(body_stmt):
                            if _mutation_base(call, _GROW_METHODS):
                                return True
            aug = _augassign_base(stmt)
            if aug and aug[1] == "shrink":  # shrink = "-=", the relaxation direction
                mutated_so_far.add(aug[0])
    return False


def _all_bodies(node: ast.AST):
    """Yields every statement-list body/orelse block reachable from `node`
    (including nested function defs), so a pattern can be checked against
    each block's *siblings* without caring which specific construct
    (function body, if-branch, for-loop body, ...) contains it."""
    body = getattr(node, "body", None)
    if isinstance(body, list):
        yield body
    orelse = getattr(node, "orelse", None)
    if isinstance(orelse, list):
        yield orelse
    for child in ast.iter_child_nodes(node):
        yield from _all_bodies(child)


def _augassign_base(stmt: ast.stmt) -> tuple[str, str] | None:
    """If `stmt` is `X += k` or `X -= k` where X is a bare Name or a
    Subscript, returns (base_name, "grow"/"shrink") -- the counter-based
    "choose, explore, un-choose" idiom (e.g. letter-tile-possibilities:
    `cnt[i] -= 1; ans += dfs(cnt); cnt[i] += 1`), which is structurally the
    same backtracking bracket as the append/pop container idiom but wasn't
    caught by _mutation_base at all (an AugAssign is its own statement, not
    a Call). Subscript targets use the subscripted container's name, same
    approximation _touched_names_in already makes elsewhere (not tracking
    *which* key, just that the container was touched)."""
    if not isinstance(stmt, ast.AugAssign):
        return None
    if isinstance(stmt.target, ast.Name):
        base = stmt.target.id
    elif isinstance(stmt.target, ast.Subscript):
        base = _name_of(stmt.target.value)
    else:
        return None
    if not base:
        return None
    if isinstance(stmt.op, ast.Add):
        return base, "grow"
    if isinstance(stmt.op, ast.Sub):
        return base, "shrink"
    return None


def _detect_backtrack(fn) -> bool:
    """RECURSION_STYLE=backtrack: some sibling-statement block contains a
    grow-mutation and a shrink-mutation of the SAME structure, plus a
    recursive call to `fn` -- the "choose, explore, un-choose" idiom, in
    either of two forms: container append/pop (subsets: `dfs(i+1);
    t.append(nums[i]); dfs(i+1); t.pop()`) or counter increment/decrement
    (letter-tile-possibilities: `cnt[i] -= 1; ans += dfs(cnt); cnt[i] +=
    1`). Order isn't enforced (some solutions choose-then-recurse, others
    recurse-then-choose) since presence of all three in one block is
    already a specific, low-false-positive signal on its own."""
    fn_name = fn.name
    for block in _all_bodies(fn):
        grow_bases, shrink_bases = set(), set()
        for stmt in block:
            if isinstance(stmt, ast.Expr):
                g = _mutation_base(stmt.value, _GROW_METHODS)
                if g:
                    grow_bases.add(g)
                s = _mutation_base(stmt.value, _SHRINK_METHODS)
                if s:
                    shrink_bases.add(s)
            aug = _augassign_base(stmt)
            if aug:
                base, kind = aug
                (grow_bases if kind == "grow" else shrink_bases).add(base)
        has_recursive_call = any(_is_self_call(n, fn_name) for stmt in block for n in ast.walk(stmt))
        if has_recursive_call and (grow_bases & shrink_bases):
            return True
    return False


def _assigned_names_before(tree: ast.AST, before_lineno: int) -> set[str]:
    names = set()
    for n in ast.walk(tree):
        lineno = getattr(n, "lineno", None)
        if lineno is None or lineno >= before_lineno:
            continue
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            names.update(name for name, _ in _assign_pairs(n))
    return names


def _latest_prior_value(tree: ast.AST, name: str, before_lineno: int) -> ast.AST | None:
    """The value expression of the last assignment to `name` with
    lineno < before_lineno -- used as a proxy for "what this variable looked
    like coming into the loop", to classify its role."""
    best_value, best_lineno = None, -1
    for n in ast.walk(tree):
        lineno = getattr(n, "lineno", None)
        if lineno is None or lineno >= before_lineno or lineno <= best_lineno:
            continue
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            for pair_name, value in _assign_pairs(n):
                if pair_name == name:
                    best_value, best_lineno = value, lineno
    return best_value


_CONTAINER_FACTORY_ROLES = {
    "list": "list", "dict": "hashmap", "set": "set", "defaultdict": "hashmap",
    "deque": "deque", "Counter": "hashmap", "SortedList": "ordered-container",
    "SortedSet": "ordered-container", "SortedDict": "ordered-container",
}


def _classify_role(value: ast.AST | None) -> str:
    """Classifies a variable's role from the shape of its initializing
    expression: a container factory/literal (list/dict/set/deque/...), a
    fill-value list (`[0] * n`), or anything else treated as a scalar
    (numbers, bools, strings, `float('inf')`, `-1`, ...)."""
    if value is None:
        return "unknown"
    if isinstance(value, ast.List):
        return "list"
    if isinstance(value, ast.Dict):
        return "hashmap"
    if isinstance(value, ast.Set):
        return "set"
    if isinstance(value, ast.Call):
        fname = _name_of(value.func) or (value.func.attr if isinstance(value.func, ast.Attribute) else None)
        if fname in _CONTAINER_FACTORY_ROLES:
            return _CONTAINER_FACTORY_ROLES[fname]
        return "scalar"  # len(...), max(...), min(...), float(...), etc.
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
        if isinstance(value.left, ast.List) or isinstance(value.right, ast.List):
            return "list"
        return "scalar"
    if isinstance(value, (ast.Constant, ast.UnaryOp, ast.BinOp, ast.Compare, ast.BoolOp)):
        return "scalar"
    return "unknown"


def _monotonic_direction(tree: ast.AST, name: str) -> str | None:
    """Scans the whole function for self-increments/decrements of `name`
    (reusing motif_detectors' delta extraction, which already handles
    `x += 1`, `x = x + 1`, and tuple-unpacking updates like `i, j = i+1,
    j-1`). Returns "increasing"/"decreasing" if every observed delta has the
    same sign, else None (mixed directions, or no simple-offset updates
    found at all)."""
    directions = set()
    for node in ast.walk(tree):
        deltas = _md_var_deltas(node)
        if name in deltas:
            directions.add(1 if deltas[name] > 0 else -1)
    if directions == {1}:
        return "increasing"
    if directions == {-1}:
        return "decreasing"
    return None


def _bucket(n: int, cap: int) -> str:
    """Buckets a count into "0", "1", ..., "{cap}+" so small numeric features
    become comparable categorical tags (exact small values are meaningful,
    e.g. lookback 2 vs 3, but past a point it's just "several")."""
    return str(n) if n < cap else f"{cap}+"


def to_tag_set(features: StructuralFeatures) -> set[str]:
    """
    Converts a StructuralFeatures instance into a set of string tags, so
    structural similarity can be computed with the same IDF-weighted
    Jaccard/Dice machinery already used for LeetCode's own tags
    (categorical_similarity.tag_similarity_matrix) -- rather than a new
    dense-vector/cosine pipeline. This matters for two reasons: it's direct
    reuse of code already trusted, and IDF weighting automatically
    down-weights near-universal tags (e.g. "loops:1") and up-weights rare,
    genuinely diagnostic ones (e.g. "lookback:3", "index_rel:converging")
    with no manual per-feature weight tuning -- exactly the rarity-awareness
    the mechanism-embedding signal was found to lack.

    Name-keyed fields (loop_carried_state, monotonic_vars) are reduced to
    their *shape* before tagging -- variable names aren't comparable across
    problems (problem A's `left` and problem B's `l` are the same concept,
    different strings), only counts and directions are.
    """
    tags = set()
    tags.add(f"loops:{_bucket(features.num_loops, 3)}")
    tags.add(f"nesting:{_bucket(features.max_loop_nesting_depth, 2)}")
    tags.update(f"nesting_pair:{p}" for p in features.loop_nesting_pairs)
    tags.update(f"for_style:{s}" for s in features.for_loop_styles)
    if features.branch_inside_loop:
        tags.add("has_branch_in_loop")
    tags.add(f"branches:{_bucket(features.num_branches_in_loop, 2)}")
    if features.early_exit_in_loop:
        tags.add("has_early_exit")
    if features.recursion:
        tags.add("has_recursion")
    if features.memoization:
        tags.add("has_memoization")
    if features.recurrence_lookback is not None:
        tags.add(f"lookback:{features.recurrence_lookback}")
    if features.recurrence_combinator is not None:
        tags.add(f"combinator:{features.recurrence_combinator}")
    if features.recursion_style is not None:
        tags.add(f"recursion_style:{features.recursion_style}")
    if features.has_repairs:
        tags.add("has_repairs")
    if features.has_grows:
        tags.add("has_grows")

    tags.add(f"carried_count:{_bucket(len(features.loop_carried_state), 3)}")
    tags.update(f"mutable:{m}" for m in features.mutable_structures)

    directions = set(features.monotonic_vars.values())
    tags.add(f"monotonic_count:{_bucket(len(features.monotonic_vars), 2)}")
    if directions:
        tags.add(f"monotonic_dirs:{'+'.join(sorted(directions))}")
    tags.update(f"index_rel:{r}" for r in features.index_relationship)

    return tags


def is_usable_solution_code(code: str) -> bool:
    """True if `code` parses and contains at least one function/method def.
    Guards against a real data-quality issue found in
    official_editorial_code_cache.pkl: a handful of entries (4/311 checked)
    are incomplete prose fragments scraped from editorial text -- a bare
    sequence of top-level statements with no function wrapper at all (e.g.
    add-bold-tag-in-string, subsets, reorder-list), not a real solution.
    Extracting features from these produces meaningless tags (a random
    prose fragment isn't "this problem's structure"), so callers should
    check this before preferring cached editorial code over the dataset's
    own `code` column, rather than trusting cache presence alone."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))


def extract_features(code: str) -> StructuralFeatures:
    tree = ast.parse(code)
    features = StructuralFeatures()

    loops = list(_walk_loops(tree))
    features.num_loops = len(loops)
    features.max_loop_nesting_depth = max((depth for _, depth, _ in loops), default=0)
    features.loop_nesting_pairs = {
        f"{parent_type}>{_loop_type(node)}" for node, depth, parent_type in loops if parent_type
    }
    features.for_loop_styles = {
        _for_loop_style(node) for node, _, _ in loops if isinstance(node, ast.For)
    }
    if any(isinstance(node, _COMPREHENSION_TYPES) for node, _, _ in loops):
        features.for_loop_styles.add("comprehension")

    if_nodes_in_loops = []
    early_exit_found = False
    comprehension_filter_count = 0
    for node, _, _ in loops:
        if_nodes_in_loops.extend(n for n in ast.walk(node) if isinstance(n, ast.If))
        if _contains(node, (ast.Return, ast.Break)):
            early_exit_found = True
        if isinstance(node, _COMPREHENSION_TYPES):
            # a comprehension's filter clauses (`for x in a if cond`) are its
            # branches, but they're plain expr nodes in gen.ifs, not ast.If
            # statements, so the ast.walk(node) scan above never finds them
            comprehension_filter_count += sum(len(gen.ifs) for gen in node.generators)
    features.branch_inside_loop = bool(if_nodes_in_loops) or comprehension_filter_count > 0
    features.num_branches_in_loop = _count_distinct(if_nodes_in_loops) + comprehension_filter_count
    features.early_exit_in_loop = early_exit_found

    features.recursion = _detect_recursion(tree)
    features.memoization = _detect_memoization(tree)

    lookbacks = []
    array_form_combinator = None
    for node, _, _ in loops:
        if not isinstance(node, ast.For):
            continue
        result = _recurrence_lookback_in_loop(node)
        if result is not None:
            lb, combinator = result
            lookbacks.append(lb)
            if combinator is not None:
                array_form_combinator = combinator
    # rolling-scalar-variable form: check every loop's own body directly
    # (the rotation is a flat sequence of statements, not something nested
    # deeper that ast.walk would need to find). Both idioms only ever match
    # a pure sum-of-prior-terms update (see _sum_of_names), so finding one
    # means the combinator is "sum" -- there's no separate detection step
    # for it, unlike the explicit-array and recursive cases.
    found_rolling_recurrence = False
    for node, _, _ in loops:
        body = getattr(node, "body", None)
        if body:
            lb = _rolling_recurrence_lookback(body)
            if lb is not None:
                lookbacks.append(lb)
                found_rolling_recurrence = True
            for stmt in body:
                lb2 = _tuple_rolling_recurrence_lookback(stmt)
                if lb2 is not None:
                    lookbacks.append(lb2)
                    found_rolling_recurrence = True
    features.recurrence_lookback = max(lookbacks) if lookbacks else None

    # recurrence_combinator: how a recurrence's prior-state terms combine.
    # Same lookback distance can mean a "counting" recurrence (sum, e.g.
    # climbing-stairs) or an "optimization" recurrence (max/min, e.g.
    # house-robber, or edit-distance's explicit-array `min(...)+1`) --
    # lookback alone conflates these (see scratch_structural_eval_harness.py's
    # n-th-tribonacci-number/stone-game-iii finding). Rolling-scalar form is
    # checked first (cheap, and by construction always "sum"), then the
    # explicit-array form's own combinator, then recursive self-combining
    # form, which covers memoized-recursion cases (house-robber,
    # stone-game-iii, tallest-billboard) that lookback detection never
    # reaches at all (recurrence_lookback stays None for these).
    if found_rolling_recurrence:
        features.recurrence_combinator = "sum"
    elif array_form_combinator is not None:
        features.recurrence_combinator = array_form_combinator
    else:
        for fn in _function_defs(tree):
            combinator = _recursive_combinator(fn)
            if combinator is not None:
                features.recurrence_combinator = combinator
                break

    # recursion_style: how a recursive function delivers its result --
    # RETURN_VALUE_COMBINED ("combined", via recurrence_combinator above),
    # SIDE_EFFECT_DRIVEN backtracking ("backtrack", checked first since it's
    # the more specific signal), or pure TRAVERSAL_ONLY ("traversal",
    # e.g. flood-fill counting connected components, no combinator and no
    # backtrack bracket -- the catch-all). A flat has_recursion boolean
    # conflates all three, which is why, e.g., number-of-islands and
    # subsets and house-robber all just show up as "has_recursion" today.
    if features.recursion:
        if any(_detect_backtrack(fn) for fn in _function_defs(tree)):
            features.recursion_style = "backtrack"
        elif features.recurrence_combinator is not None:
            features.recursion_style = "combined"
        else:
            features.recursion_style = "traversal"

    features.has_repairs = _detect_repairs(tree)
    features.has_grows = _detect_grows(tree)

    # phase 2: loop-carried state. A name counts as loop-carried for a given
    # loop if it's assigned somewhere inside that loop AND already had an
    # assignment earlier in the function (before the loop starts) -- the
    # signature of an accumulator/pointer/container that persists across
    # iterations, as opposed to a purely loop-local temporary introduced
    # fresh inside the loop with no life before it.
    carried_first_loop_lineno: dict[str, int] = {}
    for node, _, _ in loops:
        inside = _touched_names_in(node)
        before = _assigned_names_before(tree, node.lineno)
        for name in (inside & before):
            if name not in carried_first_loop_lineno or node.lineno < carried_first_loop_lineno[name]:
                carried_first_loop_lineno[name] = node.lineno
    all_carried = set(carried_first_loop_lineno)
    features.loop_carried_state = all_carried

    # role is classified from the variable's value right before the
    # EARLIEST loop it's carried into (its true initializer shape), not
    # whatever it's last reassigned to -- e.g. binary search's `left` starts
    # as `0` (scalar) even though it's later reassigned to `mid + 1` inside
    # the loop.
    roles = {name: _classify_role(_latest_prior_value(tree, name, carried_first_loop_lineno[name])) for name in all_carried}
    features.mutable_structures = {role for role in roles.values() if role not in ("scalar", "unknown")}

    # Also tag any container built anywhere in the function, whether or not
    # it's later mutated inside a loop -- e.g. most-frequent-even-element's
    # `cnt = Counter(...)` is read-only inside its loop (`.items()`, never
    # written to), so it never enters loop_carried_state above and the
    # solution got NO signal at all for using a hashmap. A "build a
    # structure, then scan it read-only" family (frequency-map lookups,
    # e.g. also sort-characters-by-frequency, first-unique-character-in-a-
    # string) needs this to be visible on the same mutable:X axis as an
    # actively-mutated one, even though the two usages differ (this doesn't
    # distinguish read-only from mutated -- a coarser but still real signal
    # improvement over tagging nothing).
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for _, value in _assign_pairs(node):
                role = _classify_role(value)
                if role not in ("scalar", "unknown"):
                    features.mutable_structures.add(role)

    scalar_names = [name for name, role in roles.items() if role == "scalar"]
    monotonic_vars = {}
    for name in scalar_names:
        direction = _monotonic_direction(tree, name)
        if direction:
            monotonic_vars[name] = direction
    features.monotonic_vars = monotonic_vars

    directions_seen = list(monotonic_vars.values())
    for i in range(len(directions_seen)):
        for j in range(i + 1, len(directions_seen)):
            if directions_seen[i] == directions_seen[j]:
                features.index_relationship.add("parallel")
            else:
                features.index_relationship.add("converging")

    return features
