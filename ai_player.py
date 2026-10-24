"""
ai_player.py - the opponent that shoots at the other fleet. Five selectable "brains":

    random  : fires at a random unknown cell                               (baseline)
    hunt    : classic hunt/target with checkerboard parity                 (baseline)
    logic   : pure knowledge-based agent. Forward chaining + resolution decide the
              shot (forced cells first, then likely cells); it explains every step like an
              expert system.
    density : counts every legal ship placement that covers each cell (hits weigh far more)
              and fires at the top cell. A fast heuristic.
    csp     : the full agent. Each turn it
                1. runs forward chaining on its knowledge base
                2. solves the fleet-placement CSP, counting layouts
                3. turns the layouts into a probability heatmap
                4. optionally re-weights using what it learned about the opponent:
                      tree   - decision tree odds on cells (an odds multiplier, so it can re-order
                         middling cells but never override a cell the CSP proves)
                      kmeans - layouts matching the opponent's K-means style count more
                      replay - layouts the opponent used before are injected as candidates
                5. fires at the best cell and explains why

Agent view: percept = (cell, hit/miss/sunk); state = KnowledgeBase;
action = cell to fire at; performance measure = fewest shots to sink the fleet.
The environment is partially observable, sequential, static, discrete and adversarial.
"""
import random
import time

from board import SIZE, idx, cell_name, neighbors, placements
from knowledge_base import KnowledgeBase, UNKNOWN, HIT, MISS, SUNK
from probability import estimate

BRAINS = ["random", "hunt", "logic", "density", "csp"]
LEARNERS = ["tree", "kmeans", "replay"]

BRAIN_TEXT = {
    "random": "Fires at a random cell. The baseline to beat.",
    "hunt": "Checkerboard search, then chases any hit it finds (hunt / target).",
    "logic": "Knowledge base + forward chaining + resolution. Explains every shot like an expert system.",
    "density": "Counts the legal ship placements covering each cell and fires at the top cell.",
    "csp": "Constraint solver over every possible fleet layout, turned into a probability heatmap.",
}
LEARN_TEXT = {
    "tree": "Decision tree: favours cells where the opponent often hides ships.",
    "kmeans": "K-means style: layouts that match the opponent's style count more.",
    "replay": "Replay detector: remembers layouts the opponent has repeated.",
}


def _posterior(p, odds):
    """Re-weight a probability by an ODDS multiplier (a likelihood ratio).

    posterior = p*odds / (p*odds + (1-p)).  Unlike plain p*odds, a cell the constraints prove
    holds a ship (p = 1) stays certain, and a cell that is impossible (p = 0) stays impossible,
    so what the AI learned about you can re-order middling cells but never override evidence.
    """
    if odds == 1.0 or p <= 0.0 or p >= 1.0:
        return p
    return p * odds / (p * odds + (1.0 - p))


def _forb_hit(kb):
    forb = hit = 0
    for r in range(SIZE):
        for c in range(SIZE):
            s = kb.state[r][c]
            if s in (MISS, SUNK):
                forb |= 1 << idx(r, c)
            elif s == HIT:
                hit |= 1 << idx(r, c)
    return forb, hit


class AIPlayer:
    def __init__(self, strategy="csp", rng=None, profile=None, samples=200, learn=None):
        if strategy not in BRAINS:
            raise ValueError(f"unknown brain {strategy!r}")
        self.strategy = strategy
        self.rng = rng or random.Random()
        self.profile = profile
        self.samples = samples
        if learn is None:                       # old behaviour: tree bias when a profile is given
            learn = {"tree"} if profile is not None else set()
        self.learn = set(learn) if profile is not None else set()
        self.kb = KnowledgeBase()
        self.heat = None          # last heatmap (for the UI)
        self.heat_exact = False
        self.heat_kind = None     # "prob" or "relative"
        self.last = None

    # ---- percept ---------------------------------------------------------
    def observe(self, r, c, result, sunk_ship=None):
        self.kb.tell(r, c, result, sunk_ship)

    # ---- action ----------------------------------------------------------
    def choose(self):
        d = getattr(self, "_choose_" + self.strategy)()
        d["brain"] = self.strategy
        d["heat"], d["heat_kind"] = self.heat, self.heat_kind
        self.last = d
        return d

    def _decision(self, cell, mode, lines, **extra):
        d = {"cell": cell, "mode": mode, "lines": lines, "p": None, "n": 0,
             "exact": False, "nodes": 0, "ms": 0.0, "dead": 0, "forced": 0, "top": [],
             "detail": [], "learn_notes": [], "table": [], "steps": []}
        d.update(extra)
        return d

    # ---- random ----------------------------------------------------------
    def _choose_random(self):
        self.heat = self.heat_kind = None
        cell = self.rng.choice(self.kb.unknown_cells())
        return self._decision(cell, "RANDOM", [f"Fired at {cell_name(*cell)} at random."])

    # ---- hunt / target ---------------------------------------------------
    def _choose_hunt(self, note=None):
        kb = self.kb
        self.heat = self.heat_kind = None
        targets = []
        for (r, c) in kb.hit_cells():
            for n in neighbors(r, c):
                if kb.state[n[0]][n[1]] == UNKNOWN:
                    br, bc = 2 * n[0] - r, 2 * n[1] - c           # cell beyond n, in line
                    aligned = 0 <= br < SIZE and 0 <= bc < SIZE and kb.state[br][bc] == HIT
                    targets.append((0 if aligned else 1, n))
        if targets:
            targets.sort()
            best = [t for t in targets if t[0] == targets[0][0]]
            cell = self.rng.choice(best)[1]
            mode = "TARGET"
            lines = [f"Fired at {cell_name(*cell)}: next to an unsunk hit."]
        else:
            unknown = kb.unknown_cells()
            parity = [u for u in unknown if (u[0] + u[1]) % 2 == 0] or unknown
            cell = self.rng.choice(parity)
            mode = "HUNT"
            lines = [f"Fired at {cell_name(*cell)}: searching on a checkerboard pattern."]
        if note:
            lines.insert(0, note)
        return self._decision(cell, mode, lines)

    # ---- logic (rules and resolution only) ---------------------------------------------
    def _choose_logic(self):
        kb = self.kb
        t0 = time.perf_counter()
        self.heat = self.heat_kind = None
        kb.forward_chain()
        dead = kb.count("Dead")
        forced = sorted(f[1] for f in kb.facts if f[0] == "Forced")
        likely = sorted(f[1] for f in kb.facts if f[0] == "Likely")
        detail = [f"{why}" for f, why in kb.facts.items() if f[0] in ("Forced", "Likely", "Dead")][:40]
        base = dict(dead=dead, forced=len(forced), detail=detail, ms=(time.perf_counter() - t0) * 1000,
                    steps=[f"1. Forward chaining over rules R1-R6 found {dead} dead, {len(forced)} forced, "
                           f"{len(likely)} likely cell(s).",
                           "2. A cell PROVED by resolution is fired first.",
                           "3. Otherwise a cell that rule R6 says is likely; then a cell next to an unsunk hit.",
                           "4. With nothing known, search on a checkerboard."])
        if forced:
            cell = forced[0]
            lines = [f"Fired at {cell_name(*cell)}: PROVED by resolution that a ship is here."]
            lines += kb.explain(cell)[:2]
            if cell in kb.proofs:
                lines.append("Proof: " + "; ".join(kb.proofs[cell]))
            return self._decision(cell, "TARGET", lines, **base)
        if likely:
            cell = self.rng.choice(likely)
            lines = [f"Fired at {cell_name(*cell)}: rule R6 says the ship probably continues here."]
            lines += kb.explain(cell)[:1]
            return self._decision(cell, "TARGET", lines, **base)
        hits = kb.hit_cells()
        targets = [n for (r, c) in hits for n in neighbors(r, c)
                   if kb.state[n[0]][n[1]] == UNKNOWN and ("Dead", n) not in kb.facts]
        if targets:
            cell = self.rng.choice(targets)
            lines = [f"Fired at {cell_name(*cell)}: no rule proves a cell, so probe next to an unsunk hit."]
            return self._decision(cell, "TARGET", lines, **base)
        free = [u for u in kb.unknown_cells() if ("Dead", u) not in kb.facts] or kb.unknown_cells()
        parity = [u for u in free if (u[0] + u[1]) % 2 == 0] or free
        cell = self.rng.choice(parity)
        lines = [f"Fired at {cell_name(*cell)}: nothing is known, search on a checkerboard."]
        if dead:
            lines.append(f"Rule R4 already ruled out {dead} cell(s): no remaining ship fits there.")
        return self._decision(cell, "HUNT", lines, **base)

    # ---- density heuristic -------------------------------------------------
    def _choose_density(self):
        kb = self.kb
        t0 = time.perf_counter()
        forb, hit = _forb_hit(kb)
        score = [0.0] * (SIZE * SIZE)
        for _, length in kb.remaining:
            for mask, cells, _r, _c, _h in placements(length):
                if mask & forb:
                    continue
                w = 1 + 30 * (mask & hit).bit_count()      # placements covering hits count far more
                for (r, c) in cells:
                    score[r * SIZE + c] += w
        unknown = kb.unknown_cells()
        peak = max(score[idx(r, c)] for r, c in unknown)
        heat = [0.0] * (SIZE * SIZE)
        for (r, c) in unknown:
            heat[idx(r, c)] = score[idx(r, c)] / peak
        self.heat, self.heat_kind, self.heat_exact = heat, "relative", False
        best = [u for u in unknown if heat[idx(*u)] == 1.0]
        cell = self.rng.choice(best)
        mode = "TARGET" if kb.hit_cells() else "HUNT"
        ranked = sorted(unknown, key=lambda u: -heat[idx(*u)])[:3]
        lines = [f"Fired at {cell_name(*cell)}: the most legal ship placements pass through it "
                 f"(hit-covering placements weigh 31x).",
                 "Runners-up: " + ", ".join(f"{cell_name(*u)} {heat[idx(*u)]:.0%}" for u in ranked[1:])]
        top8 = sorted(unknown, key=lambda u: -heat[idx(*u)])[:8]
        return self._decision(cell, mode, lines, p=1.0, top=[(u, heat[idx(*u)]) for u in ranked],
                              ms=(time.perf_counter() - t0) * 1000,
                              table=[(u, heat[idx(*u)], 1.0, heat[idx(*u)]) for u in top8],
                              steps=["1. Count every legal placement of every remaining ship that avoids known misses.",
                                     "2. Placements that cover a hit count 31 times as much.",
                                     "3. Each cell's score = total weight of placements through it.",
                                     "4. Pick: the cell with the top score (100 = best)."])

    # ---- CSP + probability + learning --------------------------------------
    def _choose_csp(self):
        kb = self.kb
        t0 = time.perf_counter()
        prof = self.profile

        # forward chaining
        kb.forward_chain()
        dead = kb.count("Dead")
        forced = [f[1] for f in kb.facts if f[0] == "Forced"]

        # 2) hooks that change the layout weights
        weight_fn = prof.kmeans_weight_fn() if (prof and "kmeans" in self.learn) else None
        replay = prof.replay_prior() if (prof and "replay" in self.learn) else None

        # CSP search -> layouts -> heatmap
        est = estimate(kb, self.rng, n_samples=self.samples, log_weight_fn=weight_fn, replay=replay)
        if est is None:
            return self._choose_hunt(note="CSP found no consistent layout; fell back to hunt/target.")
        prob = est["prob"]
        self.heat, self.heat_exact, self.heat_kind = prob, est["exact"], "prob"
        kb.csp_facts = {}
        if est["exact"] and weight_fn is None:
            for (r, c) in kb.unknown_cells():
                p = prob[idx(r, c)]
                if p >= 0.999999:
                    kb.csp_facts[("Certain", (r, c))] = (
                        f"CSP: {cell_name(r, c)} holds a ship in all {est['n']} consistent layouts")
                elif p == 0:
                    kb.csp_facts[("Impossible", (r, c))] = (
                        f"CSP: no consistent layout uses {cell_name(r, c)}")

        # decision tree bias on cells, applied as an odds multiplier (see _posterior)
        use_tree = prof is not None and "tree" in self.learn
        cands = []
        for (r, c) in kb.unknown_cells():
            p = prob[idx(r, c)]
            b = prof.cell_bias(r, c) if use_tree else 1.0
            s = _posterior(p, b)
            cands.append((s + self.rng.random() * 1e-9, s, p, b, (r, c)))
        cands.sort(reverse=True)
        _, score, p, bias, cell = cands[0]

        # 5) explain
        hits = kb.hit_cells()
        mode = "TARGET" if hits else "HUNT"
        how = (f"counted all {est['n']} consistent fleet layouts" if est["exact"]
               else f"sampled {est['n']} consistent fleet layouts")
        lines = [f"Fired at {cell_name(*cell)}: {p:.0%} of layouts put a ship here ({how})."]
        if hits:
            lines.append("Only layouts that cover every unsunk hit ("
                         + ", ".join(cell_name(*h) for h in hits)
                         + ") count - that is a CSP constraint.")
        for w in kb.explain(cell)[:2]:
            lines.append(w)
        if cell in kb.proofs:
            lines.append("Proof: " + "; ".join(kb.proofs[cell]))
        if dead:
            lines.append(f"Logic ruled out {dead} cell(s): no remaining ship fits there.")
        notes = []
        if use_tree and abs(bias - 1.0) > 0.08:
            notes.append(f"Tree: the opponent {'often' if bias > 1 else 'rarely'} hides ships "
                         f"in cells like this (odds x{bias:.2f}, final score {score:.0%}).")
        if weight_fn is not None:
            temper = f", softened x{est['tau']:.2f} to stay reliable" if est["tau"] < 0.99 else ""
            notes.append(f"K-means: style '{prof.style_name}' - layouts that fit it count more "
                         f"(effective samples {est['ess']:.0f} of {est['n']}{temper}).")
        if replay is not None:
            if est["injected"]:
                notes.append(f"Replay: {replay['m']:.0%} of belief sits on a layout the opponent "
                             f"has used before and no shot contradicts it yet.")
            else:
                notes.append("Replay: every layout the opponent repeated has been ruled out by the shots.")
        lines += notes
        top = [(c[4], c[1]) for c in cands[:3]]
        table = [(c[4], c[2], c[3], c[1]) for c in cands[:8]]       # cell, layouts say, odds, final
        if len(top) > 1:
            lines.append("Runners-up: " + ", ".join(f"{cell_name(*t[0])} {t[1]:.0%}" for t in top[1:]))
        detail = [why for f, why in kb.facts.items() if f[0] in ("Forced", "Likely", "Dead")][:40]
        steps = [f"1. Logic: forward chaining found {dead} dead cell(s) and {len(forced)} forced cell(s).",
                 f"2. CSP: {est['n']} consistent layouts ({'exact count' if est['exact'] else 'sampled'}), "
                 f"{est['nodes']} search nodes.",
                 "3. Pick: the unknown cell with the highest final score."]
        return self._decision(cell, mode, lines, p=p, score=score, n=est["n"], exact=est["exact"],
                              nodes=est["nodes"], ms=(time.perf_counter() - t0) * 1000,
                              dead=dead, forced=len(forced), top=top, learn_notes=notes,
                              detail=detail, table=table, steps=steps)
