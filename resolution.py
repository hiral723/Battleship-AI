"""
resolution.py - a tiny propositional-logic theorem prover.

Clauses are frozensets of literals; a literal is (atom_name, is_positive).
`entails` proves a goal by *resolution refutation*:
    add NOT(goal) to the clauses and try to derive the empty clause.
"""


def negate(lit):
    return lit[0], not lit[1]


def fmt_lit(lit):
    return lit[0] if lit[1] else "~" + lit[0]


def fmt(clause):
    if not clause:
        return "FALSE"
    return " or ".join(sorted(fmt_lit(l) for l in clause))


def resolve_pair(c1, c2):
    """All resolvents of two clauses (tautologies are dropped)."""
    out = []
    for lit in c1:
        if negate(lit) in c2:
            res = (c1 - {lit}) | (c2 - {negate(lit)})
            if any(negate(l) in res for l in res):
                continue
            out.append(frozenset(res))
    return out


def entails(clauses, goal, max_clauses=400):
    """Returns (proved, proof_steps). `goal` is a literal."""
    base = [frozenset(c) for c in clauses] + [frozenset([negate(goal)])]
    known = set(base)
    parents = {}
    while True:
        items = list(known)
        new = set()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                for res in resolve_pair(items[i], items[j]):
                    if res in known or res in new:
                        continue
                    parents[res] = (items[i], items[j])
                    if not res:
                        return True, _trace(res, parents)
                    new.add(res)
        if not new:
            return False, []
        known |= new
        if len(known) > max_clauses:
            return False, []


def _trace(empty, parents):
    steps, seen = [], set()

    def visit(c):
        if c in seen or c not in parents:
            return
        seen.add(c)
        a, b = parents[c]
        visit(a)
        visit(b)
        steps.append(f"resolve [{fmt(a)}] with [{fmt(b)}] -> [{fmt(c)}]")

    visit(empty)
    return steps
