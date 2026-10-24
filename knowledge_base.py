"""
knowledge_base.py - what the AI knows about the enemy fleet.

The AI observes shots (percepts) and stores them as facts. Forward chaining then
applies rules until no new fact can be derived. Every derived fact carries a
human-readable justification, which is how the AI "explains" its moves.

Facts are tuples  (kind, (row, col))  with kinds:
    Miss, Hit, Taken     - raw observations
    NoShip, Ship         - R1 / R2
    Dead                 - R4: no remaining ship can possibly cover this cell
    Forced               - R5: proved by resolution that a ship is here
    Likely               - R6: heuristic, hits in a line suggest where it continues
    Certain, Impossible  - added by the CSP when it enumerated *all* layouts
"""
from board import SIZE, SHIPS, neighbors, cell_name
import resolution

UNKNOWN, MISS, HIT, SUNK = 0, 1, 2, 3


class KnowledgeBase:
    def __init__(self):
        self.state = [[UNKNOWN] * SIZE for _ in range(SIZE)]
        self.remaining = list(SHIPS)      # ships not yet sunk: (name, length)
        self.facts = {}                   # fact -> justification text
        self.csp_facts = {}               # filled in by the AI after CSP search
        self.proofs = {}                  # cell -> resolution proof steps

    # ---- percepts --------------------------------------------------------
    def tell(self, r, c, result, sunk_ship=None):
        """Record the outcome of a shot. A sunk ship is revealed (as in most
        digital versions), so its cells become SUNK and the ship leaves
        `remaining`."""
        if result == "miss":
            self.state[r][c] = MISS
        elif result == "hit":
            self.state[r][c] = HIT
            if sunk_ship is not None:
                for a, b in sunk_ship.cells:
                    self.state[a][b] = SUNK
                self.remaining = [s for s in self.remaining if s[0] != sunk_ship.name]

    # ---- helpers ---------------------------------------------------------
    def cells_in_state(self, wanted):
        return [(r, c) for r in range(SIZE) for c in range(SIZE) if self.state[r][c] == wanted]

    def unknown_cells(self):
        return self.cells_in_state(UNKNOWN)

    def hit_cells(self):
        return self.cells_in_state(HIT)

    def min_remaining(self):
        return min((l for _, l in self.remaining), default=99)

    def count(self, kind):
        return sum(1 for f in self.facts if f[0] == kind)

    def _dead(self, r, c):
        return ("Dead", (r, c)) in self.facts

    def _blocked(self, r, c):
        return self.state[r][c] in (MISS, SUNK) or self._dead(r, c)

    def _free(self, r, c):
        return self.state[r][c] in (UNKNOWN, HIT) and not self._dead(r, c)

    def _add(self, fact, why):
        if fact in self.facts:
            return False
        self.facts[fact] = why
        return True

    # ---- forward chaining ------------------------------------------------
    def forward_chain(self):
        """Re-derive all facts from the observations. Returns the new facts."""
        before = set(self.facts)
        self.facts, self.proofs = {}, {}
        self._observations()
        changed = True
        while changed:                      # run rules to a fixpoint
            changed = False
            changed |= self._rule_dead_cells()
            changed |= self._rule_resolution()
        self._rule_alignment()
        return [f for f in self.facts if f not in before]

    def _observations(self):
        for r in range(SIZE):
            for c in range(SIZE):
                n, s = cell_name(r, c), self.state[r][c]
                if s == MISS:
                    self._add(("Miss", (r, c)), f"observed: shot at {n} missed")
                    self._add(("NoShip", (r, c)), f"R1: Miss({n}) => ~Ship({n})")
                elif s == HIT:
                    self._add(("Hit", (r, c)), f"observed: shot at {n} hit")
                    self._add(("Ship", (r, c)), f"R2: Hit({n}) => Ship({n})")
                elif s == SUNK:
                    self._add(("Taken", (r, c)),
                              f"R3: {n} belongs to a sunk ship, so no remaining ship can use it")

    def _run(self, r, c, dr, dc):
        n = 1
        for sign in (1, -1):
            a, b = r + sign * dr, c + sign * dc
            while 0 <= a < SIZE and 0 <= b < SIZE and self._free(a, b):
                n += 1
                a += sign * dr
                b += sign * dc
        return n

    def _rule_dead_cells(self):
        """R4: if the longest free line through a cell is shorter than the
        smallest remaining ship, no ship can be there."""
        m, changed = self.min_remaining(), False
        for r in range(SIZE):
            for c in range(SIZE):
                if self.state[r][c] != UNKNOWN or self._dead(r, c):
                    continue
                best = max(self._run(r, c, 0, 1), self._run(r, c, 1, 0))
                if best < m:
                    n = cell_name(r, c)
                    changed |= self._add(
                        ("Dead", (r, c)),
                        f"R4: longest free line through {n} is {best} cells but the "
                        f"smallest remaining ship needs {m} => ~Ship({n})")
        return changed

    def _rule_resolution(self):
        """R5: a hit cell belongs to a ship of length >= 2, so at least one of
        its neighbours holds a ship:  Ship(N1) or Ship(N2) or ...
        Resolution with the known-empty neighbours proves which one."""
        changed = False
        atom = lambda cell: f"Ship({cell_name(*cell)})"
        for (r, c) in self.hit_cells():
            nbrs = list(neighbors(r, c))
            if any(self.state[a][b] == HIT for a, b in nbrs):
                continue                          # clause already satisfied
            clause = frozenset((atom(n), True) for n in nbrs)
            units = [frozenset([(atom(n), False)]) for n in nbrs if self._blocked(*n)]
            free = [n for n in nbrs if not self._blocked(*n)]
            if not units or not free:
                continue
            for n in free:
                if ("Forced", n) in self.facts:
                    continue
                proved, steps = resolution.entails([clause] + units, (atom(n), True))
                if proved:
                    changed |= self._add(
                        ("Forced", n),
                        f"R5 (resolution): hit at {cell_name(r, c)} needs a ship neighbour and "
                        f"every other neighbour is empty => Ship({cell_name(*n)})")
                    self.proofs[n] = steps
        return changed

    def _rule_alignment(self):
        """R6 (heuristic): two or more adjacent hits in a line suggest the ship
        continues at either end of that line."""
        for horizontal in (True, False):
            for fixed in range(SIZE):
                run = []
                for step in range(SIZE + 1):
                    r, c = (fixed, step) if horizontal else (step, fixed)
                    if step < SIZE and self.state[r][c] == HIT:
                        run.append((r, c))
                        continue
                    if len(run) >= 2:
                        a, b = run[0], run[-1]
                        ends = [(a[0], a[1] - 1) if horizontal else (a[0] - 1, a[1]),
                                (b[0], b[1] + 1) if horizontal else (b[0] + 1, b[1])]
                        names = ",".join(cell_name(*x) for x in run)
                        for e in ends:
                            if 0 <= e[0] < SIZE and 0 <= e[1] < SIZE and \
                                    self.state[e[0]][e[1]] == UNKNOWN and not self._dead(*e):
                                self._add(("Likely", e),
                                          f"R6 (heuristic): hits at {names} lie in a line, "
                                          f"so the ship probably continues at {cell_name(*e)}")
                    run = []

    # ---- explanations ----------------------------------------------------
    def explain(self, cell):
        """Justifications of every derived fact about one cell."""
        out = []
        for f, why in list(self.facts.items()) + list(self.csp_facts.items()):
            if f[1] == cell and f[0] not in ("Miss", "Hit", "NoShip", "Ship", "Taken"):
                out.append(why)
        return out
