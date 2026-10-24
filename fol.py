"""
fol.py - first-order logic engine: unification, backward chaining, forward chaining.

Knowledge is written as Horn clauses with VARIABLES (names starting with "?"), which are
implicitly universally quantified:

    ForcedShip(?u) :- Hit(?x), Up(?x,?u), Unknown(?u), Down(?x,?d), NoShip(?d), ...
    reads: for all x, u, d, ...: if x is a hit and u is above x and u is unexplored and all
    other neighbours of x are empty, then u must hold a ship.

Facts are built from what the AI has seen (Hit(E5), Miss(B2), Unknown(C3), Up(E5,E4) ...).
There is a pseudo-cell WALL standing for "off the board" so every cell has four neighbours.

    backward chaining : start from a goal such as  Ship(E5)?  or  ForcedShip(?c)?  and search
                        for rules/facts that prove it (like Prolog). Returns a proof tree.
    forward chaining  : apply every rule to the known facts until nothing new appears.

The first-order rules live in RULES below; the propositional resolution prover that works
with the same board is in resolution.py (rule R5 of knowledge_base.py).
"""
import re

from board import SIZE, cell_name
from knowledge_base import UNKNOWN, MISS, HIT, SUNK

DIRS = ("Up", "Down", "Left", "Right")
OPPOSITE = {"Up": "Down", "Down": "Up", "Left": "Right", "Right": "Left"}


# ---------------------------------------------------------------- terms ----
def is_var(t):
    return isinstance(t, str) and t.startswith("?")


def walk(t, th):
    while is_var(t) and t in th:
        t = th[t]
    return t


def _unify_term(a, b, th):
    a, b = walk(a, th), walk(b, th)
    if a == b:
        return th
    if is_var(a):
        out = dict(th)
        out[a] = b
        return out
    if is_var(b):
        out = dict(th)
        out[b] = a
        return out
    return None


def unify(x, y, th):
    """Most general unifier of two atoms under substitution th, or None."""
    if len(x) != len(y) or x[0] != y[0]:
        return None
    for a, b in zip(x[1:], y[1:]):
        th = _unify_term(a, b, th)
        if th is None:
            return None
    return th


def subst(atom, th):
    return (atom[0],) + tuple(walk(t, th) for t in atom[1:])


def fmt(atom):
    return f"{atom[0]}({','.join(atom[1:])})"


_ATOM = re.compile(r"^\s*(\w+)\s*\(\s*(.*?)\s*\)\s*$")


def parse_atom(text):
    m = _ATOM.match(text)
    if not m:
        raise ValueError(f"cannot read '{text}' - write it like Ship(E5) or ForcedShip(?c)")
    args = [a.strip() for a in m.group(2).split(",")] if m.group(2).strip() else []
    args = [a if a.startswith("?") else a.upper() for a in args]
    return (m.group(1),) + tuple(args)


def parse_rule(label, text):
    head, _, body = text.partition(":-")
    atoms = re.findall(r"\w+\([^)]*\)", body)
    return (label, parse_atom(head), [parse_atom(a) for a in atoms])


# ------------------------------------------------------------ rule base ----
def _rule_texts():
    rules = [
        ("R1", "NoShip(?x) :- Blocked(?x)"),
        ("R2", "NoShip(?x) :- Isolated(?x)"),
        ("R3", "Blocked(?x) :- Miss(?x)"),
        ("R4", "Blocked(?x) :- Sunk(?x)"),
        ("R5", "Ship(?x) :- Hit(?x)"),
        ("R6", "Isolated(?x) :- Unknown(?x), Up(?x,?a), Blocked(?a), Down(?x,?b), Blocked(?b), "
               "Left(?x,?c), Blocked(?c), Right(?x,?d), Blocked(?d)"),
    ]
    for d in DIRS:                                  # forced: the only open neighbour of a hit
        others = [o for o in DIRS if o != d]
        body = f"Hit(?x), {d}(?x,?u), Unknown(?u)"
        for k, o in enumerate(others):
            body += f", {o}(?x,?n{k}), NoShip(?n{k})"
        rules.append((f"R7-{d}", f"ForcedShip(?u) :- {body}"))
    for d in DIRS:                                  # likely: two hits in a line continue the line
        rules.append((f"R8-{d}", f"LikelyShip(?z) :- Hit(?x), {d}(?x,?y), Hit(?y), {d}(?y,?z), Unknown(?z)"))
    return rules


RULES = [parse_rule(label, text) for label, text in _rule_texts()]
RULE_TEXT = {label: text for label, text in _rule_texts()}
PREDICATES = {a[0] for _, h, b in RULES for a in [h] + b} | {"Miss", "Hit", "Sunk", "Unknown"}
EXAMPLES = ["Ship(E5)", "NoShip(A1)", "ForcedShip(?c)", "LikelyShip(?c)", "Hit(?c)"]


def build_facts(kb):
    """Ground facts describing the AI's current view of the board."""
    names = {UNKNOWN: "Unknown", MISS: "Miss", HIT: "Hit", SUNK: "Sunk"}
    facts = {("Blocked", "WALL")}
    for r in range(SIZE):
        for c in range(SIZE):
            n = cell_name(r, c)
            facts.add((names[kb.state[r][c]], n))
            facts.add(("Up", n, cell_name(r - 1, c) if r > 0 else "WALL"))
            facts.add(("Down", n, cell_name(r + 1, c) if r < SIZE - 1 else "WALL"))
            facts.add(("Left", n, cell_name(r, c - 1) if c > 0 else "WALL"))
            facts.add(("Right", n, cell_name(r, c + 1) if c < SIZE - 1 else "WALL"))
    return facts


# --------------------------------------------------------------- engine ----
class FOLEngine:
    def __init__(self, facts, rules=None, max_depth=14, max_steps=300000):
        self.rules = rules if rules is not None else RULES
        self.facts = set(facts)
        self.max_depth, self.max_steps = max_depth, max_steps
        self.steps = 0
        self._counter = 0
        self.by_pred, self.by_first = self._index(self.facts)

    @staticmethod
    def _index(facts):
        by_pred, by_first = {}, {}
        for f in facts:
            by_pred.setdefault(f[0], []).append(f)
            by_first.setdefault((f[0], f[1] if len(f) > 1 else None), []).append(f)
        return by_pred, by_first

    def _rename(self, head, body):
        self._counter += 1
        suffix = f"_{self._counter}"
        sub = lambda a: (a[0],) + tuple(t + suffix if is_var(t) else t for t in a[1:])
        return sub(head), [sub(a) for a in body]

    # ---- backward chaining ----
    def _candidates(self, goal, th, by_pred, by_first):
        first = walk(goal[1], th) if len(goal) > 1 else None
        if first is not None and not is_var(first):
            return by_first.get((goal[0], first), ())
        return by_pred.get(goal[0], ())

    def solve(self, goal, th, depth=0):
        if depth > self.max_depth:
            return
        self.steps += 1
        if self.steps > self.max_steps:
            return
        for fact in self._candidates(goal, th, self.by_pred, self.by_first):
            t = unify(goal, fact, th)
            if t is not None:
                yield t, (goal, "fact", [])
        for label, head, body in self.rules:
            if head[0] != goal[0] or len(head) != len(goal):
                continue
            h, b = self._rename(head, body)
            t = unify(goal, h, th)
            if t is None:
                continue
            for t2, subs in self.solve_all(b, t, depth + 1):
                yield t2, (goal, label, subs)

    def solve_all(self, goals, th, depth):
        if not goals:
            yield th, []
            return
        for t, tree in self.solve(goals[0], th, depth):
            for t2, rest in self.solve_all(goals[1:], t, depth):
                yield t2, [tree] + rest

    def ask(self, text, limit=6, compact=True):
        """Backward chaining. Returns {'goal', 'answers': [{'bindings', 'proof'}], 'steps'}."""
        goal = parse_atom(text)
        known = {p.lower(): p for p in PREDICATES}
        goal = (known.get(goal[0].lower(), goal[0]),) + goal[1:]
        if goal[0] not in PREDICATES:
            raise ValueError(f"unknown predicate {goal[0]}. Try: " + "  ".join(EXAMPLES))
        qvars = [t for t in goal[1:] if is_var(t)]
        self.steps = 0
        answers, seen = [], set()
        for th, tree in self.solve(goal, {}):
            key = tuple(walk(v, th) for v in qvars)
            if key in seen:
                continue
            seen.add(key)
            lines = []
            _proof_lines(tree, th, 0, lines, compact)
            answers.append({"bindings": {v: walk(v, th) for v in qvars}, "proof": lines})
            if len(answers) >= limit:
                break
        return {"goal": goal, "answers": answers, "steps": self.steps}

    # ---- forward chaining ----
    def forward(self):
        """Naive forward chaining to a fixpoint. Returns {derived_fact: rule_label}."""
        known, derived = set(self.facts), {}
        changed = True
        while changed:
            changed = False
            by_pred, by_first = self._index(known)
            for label, head, body in self.rules:
                for th in self._match(body, {}, by_pred, by_first):
                    new = subst(head, th)
                    if new not in known:
                        known.add(new)
                        derived[new] = label
                        changed = True
        return derived

    def _match(self, goals, th, by_pred, by_first):
        if not goals:
            yield th
            return
        for fact in self._candidates(goals[0], th, by_pred, by_first):
            t = unify(goals[0], fact, th)
            if t is not None:
                yield from self._match(goals[1:], t, by_pred, by_first)


def _proof_lines(tree, th, indent, out, compact):
    goal, label, subs = tree
    text = fmt(subst(goal, th))
    if label == "fact":
        if compact and goal[0] in DIRS:
            return
        out.append("  " * indent + f"{text}   [fact]")
        return
    out.append("  " * indent + f"{text}   [by {label}]")
    for s in subs:
        _proof_lines(s, th, indent + 1, out, compact)


def engine_for(kb):
    return FOLEngine(build_facts(kb))
