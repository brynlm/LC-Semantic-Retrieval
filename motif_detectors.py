"""
Structural (AST-based) motif detectors for common algorithmic idioms.

Unlike the LLM-generated "mechanism" abstracts, these are deterministic: the
same code always produces the same detection result, with no generation step,
no rate limits, and no risk of an LLM describing the wrong technique. The
tradeoff is coverage -- each detector only recognizes the specific syntactic
shape it's written for, so it will under-report on unusual implementations
rather than over-generalize like an LLM might.

This starts with one motif -- monotonic stack/deque -- as a prototype, tested
against real solutions we already have cached (see
scratch_test_motif_detector.py). More detectors (two-pointer convergence,
sliding window, binary search skeleton, DFS/BFS-with-visited-set,
prefix-sum accumulator, ...) should follow the same pattern: define the
syntactic signature, walk the AST for it, validate against known examples.

Usage:
    import motif_detectors as md
    matches = md.detect_monotonic_stack(code_string)
    bool(matches)  # True if the motif is present anywhere in the code
"""

import ast
from dataclasses import dataclass, field


@dataclass
class MotifMatch:
    motif: str
    container_name: str
    line: int
    details: dict = field(default_factory=dict)


def _name_of(node: ast.AST) -> str | None:
    """Returns the bare variable name a node refers to, if it's a Name."""
    return node.id if isinstance(node, ast.Name) else None


def _is_edge_index(node: ast.AST, container: str) -> bool:
    """True if `node` is exactly `container[-1]` or `container[0]` -- reads
    the container's current back/front element (as an index, or as a value if
    the container holds values directly)."""
    if not isinstance(node, ast.Subscript):
        return False
    if _name_of(node.value) != container:
        return False
    idx = node.slice
    if isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.USub):
        return isinstance(idx.operand, ast.Constant) and idx.operand.value == 1
    if isinstance(idx, ast.Constant):
        return idx.value == 0
    return False


def _outer_array_name(node: ast.AST) -> str | None:
    """If `node` is `NAME[...]` for some plain array/list name, returns NAME."""
    return _name_of(node.value) if isinstance(node, ast.Subscript) else None


def _is_container_edge_value(node: ast.AST, container: str) -> tuple[bool, str | None]:
    """
    Recognizes a reference to the *value* at the container's current
    back/front, in either of two forms a monotonic stack/deque can take:

      Case A -- the container holds values directly: the node IS
      `container[-1]` / `container[0]`. Returns (True, None): no outer array
      to cross-check against.

      Case B -- the container holds indices into a parallel values array:
      the node is `ARR[container[-1]]` / `ARR[container[0]]` (e.g.
      `nums[dq[-1]]`). Returns (True, "ARR") so the caller can verify the
      *other* side of the comparison also goes through the same array --
      which is what separates a genuine value-vs-value monotonicity check
      from an index-vs-boundary staleness check like `dq[0] < i - k`
      (comparing a stored index to a window boundary, with no value lookup
      at all -- a different motif, window-boundary eviction, not this one).

    Returns (False, None) if `node` matches neither form.
    """
    if _is_edge_index(node, container):
        return True, None
    if isinstance(node, ast.Subscript):
        outer = _outer_array_name(node)
        if outer and _is_edge_index(node.slice, container):
            return True, outer
    return False, None


def _references_array(node: ast.AST, array_name: str) -> bool:
    """True if `node` contains a subscript into `array_name` anywhere (e.g.
    checking the comparison's other side is also `array_name[...]`, such as
    `nums[i]` alongside `nums[dq[-1]]`)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and _outer_array_name(n) == array_name:
            return True
    return False


def _looks_like_index_boundary(node: ast.AST) -> bool:
    """True if `node` is pure index/offset arithmetic -- Names, Constants,
    and +/- operators only, with no Subscript or Call anywhere. This is the
    shape of a window-boundary check like `i - k` (comparing a stored index
    to how far back the window reaches), as opposed to a genuine value
    reference like `nums[i]` or a bare `num` from `for num in nums:`. Used to
    reject Case A matches (`container[-1]` compared directly) when the other
    side is really an index boundary, not a value -- e.g. `dq[0] < i - k`
    should NOT count as the monotonic-invariant comparison even though
    `dq[0]` is a valid edge-index form in isolation."""
    nodes = list(ast.walk(node))
    has_subscript_or_call = any(isinstance(n, (ast.Subscript, ast.Call)) for n in nodes)
    if has_subscript_or_call:
        return False
    # A bare Name/Constant with no arithmetic (e.g. `num` from `for num in
    # nums:`) is a plausible direct value reference, not a boundary -- only
    # an actual arithmetic combination (`i - k`, `i + 1`) reads as "offset
    # from the loop index" rather than "the current value."
    return any(isinstance(n, ast.BinOp) for n in nodes)


def _call_target_and_method(node: ast.AST) -> tuple[str, str] | None:
    """If `node` is a method call like `x.pop()` or `x.append(y)`, returns
    (x's name, method name); else None. Matches the call wherever it appears
    -- as its own bare statement, or nested as a sub-expression, since a
    popped value is very commonly used immediately (`vk = stk.pop()`,
    `vis.remove(stk.pop())`) rather than discarded on its own line."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        target = _name_of(node.func.value)
        if target:
            return target, node.func.attr
    return None


_POP_METHODS = {"pop", "popleft"}
_PUSH_METHODS = {"append", "appendleft"}


def _while_pops_container(while_node: ast.While, container: str) -> bool:
    """True if `container.pop()`/`.popleft()` appears anywhere in the
    while's body (not necessarily the only statement -- e.g. jump-game-vi's
    editorial solution has an unrelated statement between the pop and the
    loop's end)."""
    for stmt in ast.walk(while_node):
        hit = _call_target_and_method(stmt)
        if hit and hit[0] == container and hit[1] in _POP_METHODS:
            return True
    return False


def _find_append_after(body: list[ast.stmt], start_index: int, container: str) -> ast.stmt | None:
    """Scans `body[start_index:]` (statements after the while, in the same
    block) for a `container.append()`/`.appendleft()` call, returning the
    first one found."""
    for stmt in body[start_index:]:
        for node in ast.walk(stmt):
            hit = _call_target_and_method(node)
            if hit and hit[0] == container and hit[1] in _PUSH_METHODS:
                return stmt
    return None


def detect_monotonic_stack(code: str) -> list[MotifMatch]:
    """
    Detects the monotonic-stack/deque motif: a while loop whose condition
    compares the incoming value against the container's current back/front
    element, whose body pops from that container, followed later (in the
    same enclosing block) by a push back onto it. This is the structural
    signature shared by next-greater-element, sliding-window-maximum,
    largest-rectangle-in-histogram, and jump-game-vi's optimal DP transition
    -- regardless of variable names, container type (list vs deque), or
    whether other unrelated loops/pops surround it (e.g. jump-game-vi's
    window-boundary eviction pop, which has no such comparison and is
    correctly NOT flagged as part of this motif).

    Returns one MotifMatch per detected occurrence (usually one per function,
    but a function could contain more than one).
    """
    matches: list[MotifMatch] = []
    tree = ast.parse(code)

    def scan_body(body: list[ast.stmt]):
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.While):
                for node in ast.walk(stmt.test):
                    if isinstance(node, ast.Compare) and any(
                        isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)) for op in node.ops
                    ):
                        # Restricting to ordering operators specifically
                        # excludes membership/equality tests like
                        # `stk[-1] in 'tf'` or `stk[-1] == target`, which
                        # share the Compare node shape but aren't checking
                        # order/monotonicity at all.
                        sides = [node.left, *node.comparators]
                        for side_i, side in enumerate(sides):
                            other_side = sides[1 - side_i] if len(sides) == 2 else None
                            # candidate containers: every subscript anywhere
                            # in this side, in case the edge-reference isn't
                            # the side's top-level expression (e.g. it's one
                            # operand of a larger arithmetic expression)
                            for sub in ast.walk(side):
                                if not isinstance(sub, ast.Subscript):
                                    continue
                                container = _name_of(sub.value)
                                if not container:
                                    continue
                                is_edge, outer = _is_container_edge_value(sub, container)
                                if not is_edge:
                                    continue
                                # Case B requires the OTHER side to also flow
                                # through the same parallel array -- otherwise
                                # this is an index-vs-boundary staleness check
                                # (e.g. `dq[0] < i - k`), not a value comparison.
                                if outer is not None and not (other_side and _references_array(other_side, outer)):
                                    continue
                                # Case A: reject if the other side is pure
                                # index/offset arithmetic (same boundary-check
                                # shape, just without an outer array at all).
                                if outer is None and other_side is not None and _looks_like_index_boundary(other_side):
                                    continue
                                if _while_pops_container(stmt, container):
                                    push_stmt = _find_append_after(body, i + 1, container)
                                    if push_stmt is not None:
                                        matches.append(MotifMatch(
                                            motif="monotonic_stack",
                                            container_name=container,
                                            line=stmt.lineno,
                                            details={"push_line": push_stmt.lineno, "via_array": outer},
                                        ))
            for child_body_attr in ("body", "orelse", "finalbody"):
                child_body = getattr(stmt, child_body_attr, None)
                if child_body:
                    scan_body(child_body)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan_body(node.body)

    return matches


def _iter_nested_bodies(stmt: ast.stmt):
    """Yields every nested statement-list inside `stmt` -- for/while bodies,
    if/else branches, try/except/finally -- so callers can recurse uniformly
    without hardcoding which statement types have which body attributes."""
    for attr in ("body", "orelse", "finalbody"):
        child = getattr(stmt, attr, None)
        if child:
            yield child
    for handler in getattr(stmt, "handlers", None) or []:
        yield handler.body


def _extract_self_increment(target_name: str, value: ast.AST) -> int | None:
    """If `value` is exactly `target_name + 1` or `target_name - 1`, returns
    +1/-1; else None. Requires the variable to appear on the LHS of the
    arithmetic (matching `x + 1`, not e.g. `mid + 1`) so this only fires on
    genuine self-increments."""
    if not isinstance(value, ast.BinOp):
        return None
    if _name_of(value.left) != target_name:
        return None
    if not (isinstance(value.right, ast.Constant) and value.right.value == 1):
        return None
    if isinstance(value.op, ast.Add):
        return 1
    if isinstance(value.op, ast.Sub):
        return -1
    return None


def _var_deltas_in_stmt(stmt: ast.AST) -> dict[str, int]:
    """Returns {var_name: +1 or -1} for every self-increment/decrement found
    directly in `stmt` (not recursing into nested blocks) -- covers
    `x += 1`, `x -= 1`, `x = x + 1`, `x = x - 1`, and the tuple-unpacking form
    `i, j = i + 1, j - 1` that turned out to be the common idiom in real
    two-pointer code (valid-palindrome, 3sum) rather than two separate
    statements."""
    deltas: dict[str, int] = {}
    if isinstance(stmt, ast.AugAssign):
        name = _name_of(stmt.target)
        if name and isinstance(stmt.value, ast.Constant) and stmt.value.value == 1:
            if isinstance(stmt.op, ast.Add):
                deltas[name] = 1
            elif isinstance(stmt.op, ast.Sub):
                deltas[name] = -1
    elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value = stmt.targets[0], stmt.value
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts):
            pairs = list(zip(target.elts, value.elts))
        else:
            pairs = [(target, value)]
        for t, v in pairs:
            name = _name_of(t)
            if name:
                delta = _extract_self_increment(name, v)
                if delta is not None:
                    deltas[name] = delta
    return deltas


def _bound_names_from_ordering_test(test: ast.AST) -> tuple[str, str] | None:
    """If `test` is a bare Compare with an ordering operator (Lt/LtE) between
    two plain variables (e.g. `left < right`, `lo <= hi`), returns their
    names; else None. Deliberately narrow -- excludes compound conditions
    like `left < right and s[left].isalnum()` for this first pass, since
    those need more careful handling of which comparison is the "loop
    convergence" one versus an unrelated guard clause."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Lt, ast.LtE))):
        return None
    a, b = _name_of(test.left), _name_of(test.comparators[0])
    return (a, b) if (a and b) else None


def detect_two_pointer_convergence(code: str) -> list[MotifMatch]:
    """
    Detects two-pointer convergence: a while loop testing `left < right`
    (or `<=`) where one of the two variables is incremented and the other is
    decremented somewhere in the loop body -- the "moving toward each other"
    signature shared by valid-palindrome, container-with-most-water, and
    3sum's inner two-sum-on-sorted-array scan. This deliberately does NOT
    match two pointers moving in the SAME direction (a merge-style scan),
    since that's a related but distinct motif.
    """
    matches: list[MotifMatch] = []
    tree = ast.parse(code)

    def scan_body(body: list[ast.stmt]):
        for stmt in body:
            if isinstance(stmt, ast.While):
                names = _bound_names_from_ordering_test(stmt.test)
                if names:
                    left_name, right_name = names
                    deltas: dict[str, int] = {}
                    for node in ast.walk(stmt):
                        deltas.update(_var_deltas_in_stmt(node))
                    if deltas.get(left_name) == 1 and deltas.get(right_name) == -1:
                        matches.append(MotifMatch(
                            motif="two_pointer_convergence",
                            container_name=f"{left_name},{right_name}",
                            line=stmt.lineno,
                            details={},
                        ))
            for child_body in _iter_nested_bodies(stmt):
                scan_body(child_body)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan_body(node.body)

    return matches


def _strip_plus_one(node: ast.AST) -> ast.AST:
    """Strips a trailing `+ 1` if present (for the `(a+b+1)//2` mid-bias
    variant used to round up instead of down), returning the inner node
    unchanged otherwise."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and isinstance(node.right, ast.Constant) and node.right.value == 1:
        return node.left
    return node


def _is_mid_computation(value: ast.AST, bound_a: str, bound_b: str) -> bool:
    """True if `value` computes a midpoint from bound_a/bound_b: `(a+b)//2`,
    `(a+b)>>1`, or the round-up variants `(a+b+1)//2` / `(a+b+1)>>1` -- both
    divide-styles and both bias directions turned up in real code
    (binary-search used `>>1`, sqrtx used `(l+r+1)>>1`)."""
    if not isinstance(value, ast.BinOp):
        return False
    if isinstance(value.op, ast.FloorDiv) and isinstance(value.right, ast.Constant) and value.right.value == 2:
        inner = value.left
    elif isinstance(value.op, ast.RShift) and isinstance(value.right, ast.Constant) and value.right.value == 1:
        inner = value.left
    else:
        return False
    inner = _strip_plus_one(inner)
    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add):
        return {_name_of(inner.left), _name_of(inner.right)} == {bound_a, bound_b}
    return False


def _references_mid(value: ast.AST, mid_name: str) -> bool:
    """True if `value` is `mid` itself, or `mid + const`/`mid - const` (the
    `lo = mid + 1` / `hi = mid - 1` narrowing step)."""
    if _name_of(value) == mid_name:
        return True
    if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub)):
        return _name_of(value.left) == mid_name and isinstance(value.right, ast.Constant)
    return False


def detect_binary_search(code: str) -> list[MotifMatch]:
    """
    Detects the binary-search skeleton: a while loop testing two bounds
    (`lo <= hi` / `lo < hi`), a midpoint computed from them inside the loop,
    and at least one of the bounds later reassigned based on that midpoint --
    the invariant-halving structure shared by textbook array search and
    "binary search on the answer" problems alike (capacity-to-ship-packages,
    sqrtx), regardless of whether the divide is `//2` or `>>1`.
    """
    matches: list[MotifMatch] = []
    tree = ast.parse(code)

    def scan_body(body: list[ast.stmt]):
        for stmt in body:
            if isinstance(stmt, ast.While):
                names = _bound_names_from_ordering_test(stmt.test)
                if names:
                    bound_a, bound_b = names
                    mid_name = None
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Assign) and len(node.targets) == 1:
                            name = _name_of(node.targets[0])
                            if name and _is_mid_computation(node.value, bound_a, bound_b):
                                mid_name = name
                                break
                    if mid_name:
                        narrows = False
                        for node in ast.walk(stmt):
                            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                                target_name = _name_of(node.targets[0])
                                if target_name in (bound_a, bound_b) and _references_mid(node.value, mid_name):
                                    narrows = True
                                    break
                        if narrows:
                            matches.append(MotifMatch(
                                motif="binary_search",
                                container_name=f"{bound_a},{bound_b}",
                                line=stmt.lineno,
                                details={"mid": mid_name},
                            ))
            for child_body in _iter_nested_bodies(stmt):
                scan_body(child_body)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan_body(node.body)

    return matches
