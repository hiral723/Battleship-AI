"""
csp_solver.py - the enemy fleet as a Constraint Satisfaction Problem.

Variables : one per ship that is still afloat (Carrier, Battleship, ...)
Domains   : every placement (position + orientation) of that ship that does not
            touch a cell known to be empty (miss / sunk ship)
Constraints:
    1. No two ships overlap.
    2. Every cell we have HIT (on a ship that is not sunk yet) must be covered
       by some ship.
    3. A ship that is still afloat has at least one cell we have not hit yet
       (otherwise it would already have been sunk).

Search: backtracking with
    * MRV variable ordering (fewest legal placements first),
    * value ordering (placements that cover still-uncovered hits go first),
    * forward checking (after each assignment, drop overlapping placements from
      the other domains; an empty domain means backtrack),
    * coverage pruning (a hit that no remaining placement can cover => backtrack).

Placements are Python ints used as 100-bit masks, so overlap tests are one `&`.
Every solution is returned as (occupied_mask, per_ship_masks) so callers can compute
layout features (see layout.py) as well as cell counts.
"""
import random

from board import SIZE, placements, idx
from knowledge_base import MISS, SUNK, HIT


class FleetCSP:
    def __init__(self, kb, rng=None):
        self.rng = rng or random.Random()
        self.vars = list(kb.remaining)
        forbidden = hit = 0
        for r in range(SIZE):
            for c in range(SIZE):
                bit = 1 << idx(r, c)
                s = kb.state[r][c]
                if s in (MISS, SUNK):
                    forbidden |= bit
                elif s == HIT:
                    hit |= bit
        self.forbidden = forbidden
        self.hit_mask = hit
        # Domain filtering: no placement may touch a known-empty cell, and (constraint 3)
        # no ship that is still afloat can lie entirely on cells we already hit -
        # it would have been announced as sunk.
        self.domains0 = [[p[0] for p in placements(length)
                          if not (p[0] & forbidden) and (p[0] & ~hit)]
                         for _, length in self.vars]
        self.domain_sets = [set(d) for d in self.domains0]
        self.nodes = 0
        self.node_limit = 0
        self.aborted = False

    # ---- core backtracking search ---------------------------------------
    def _dfs(self, doms, occ, todo, out, max_out, randomize, assign):
        if self.aborted or len(out) >= max_out:
            return
        self.nodes += 1
        if self.nodes > self.node_limit:
            self.aborted = True
            return
        need = self.hit_mask & ~occ              # hits not covered yet
        if not todo:
            if not need:
                out.append((occ, tuple(assign[i] for i in range(len(self.vars)))))
            return
        if need:                                  # coverage pruning
            reach = 0
            for v in todo:
                for m in doms[v]:
                    reach |= m
            if need & ~reach:
                return
        # MRV: choose the variable with the smallest domain
        if randomize:
            v = min(todo, key=lambda x: (len(doms[x]), self.rng.random()))
        else:
            v = min(todo, key=lambda x: len(doms[x]))
        rest = [u for u in todo if u != v]
        values = list(doms[v])
        if randomize:
            self.rng.shuffle(values)
        if need:                                  # value ordering heuristic
            values.sort(key=lambda m: -(m & need).bit_count())
        for m in values:
            new_occ = occ | m
            new_doms = {u: [x for x in doms[u] if not (x & new_occ)] for u in rest}
            if any(not d for d in new_doms.values()):      # forward checking
                continue
            assign[v] = m
            self._dfs(new_doms, new_occ, rest, out, max_out, randomize, assign)
            if self.aborted or len(out) >= max_out:
                return
        assign.pop(v, None)

    # ---- public API ------------------------------------------------------
    def enumerate_exact(self, max_solutions=1500, node_limit=30000):
        """List every consistent layout as (occupied_mask, per_ship_masks), or None if
        there are too many to enumerate."""
        if not self.vars:
            return []
        out = []
        self.aborted = False
        self.node_limit = self.nodes + node_limit
        doms = {i: d for i, d in enumerate(self.domains0)}
        self._dfs(doms, 0, list(range(len(self.vars))), out, max_solutions + 1, False, {})
        if self.aborted or len(out) > max_solutions:
            return None
        return out

    def sample(self, n, node_limit=400, max_fail=8):
        """Draw up to n random consistent layouts (randomised restarts)."""
        out, fails = [], 0
        for _ in range(n):
            one = []
            self.aborted = False
            self.node_limit = self.nodes + node_limit
            doms = {i: d for i, d in enumerate(self.domains0)}
            self._dfs(doms, 0, list(range(len(self.vars))), one, 1, True, {})
            if one:
                out.append(one[0])
            else:
                fails += 1
                if fails >= max_fail:
                    break
        return out

    def consistent(self, masks_by_name):
        """Is a complete layout (dict ship name -> mask) consistent with what we know?
        Used by the replay detector. Returns (occupied, masks) or None."""
        occ = 0
        for i, (name, _) in enumerate(self.vars):
            m = masks_by_name.get(name)
            if m is None or m not in self.domain_sets[i] or (m & occ):
                return None
            occ |= m
        if self.hit_mask & ~occ:
            return None
        return occ, tuple(masks_by_name[n] for n, _ in self.vars)
