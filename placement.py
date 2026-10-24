"""
placement.py - three ways for the AI to hide its fleet.

All of them minimise the same objective (lower = harder to find):

    sum of "player shoots here" weights over every ship cell
  + a penalty for every pair of ships that touch (touching ships are easier to find)

    random   : a random legal fleet (baseline)
    hill     : hill climbing with sideways moves and random restarts. Plain hill climbing
               gets stuck in local optima and on plateaus - the restarts are the fix.
    genetic  : genetic algorithm - a population of fleets, tournament selection,
               uniform crossover (each ship comes from one parent), mutation (move one ship).

All of them get the same budget of about 2500 objective evaluations so they can be compared
fairly. Every method returns (fleet, info); fleet is a list of
(name, length, row, col, horizontal) and info has "best", "trace" (best score so far after
each evaluation) and a ready-made "text" for the UI.
"""
import math
import random

from board import SHIPS, placements, neighbors

BUDGET = 2500


def _score(fleet, weights, touch_penalty):
    total, owner = 0.0, {}
    for i, (_, cells, _) in enumerate(fleet):
        for (r, c) in cells:
            total += weights[r][c]
            owner[(r, c)] = i
    touching = set()
    for (r, c), i in owner.items():
        for n in neighbors(r, c):
            j = owner.get(n)
            if j is not None and j != i:
                touching.add((min(i, j), max(i, j)))
    return total + touch_penalty * len(touching)


def _legal_random(length, used, rng):
    while True:
        p = rng.choice(placements(length))
        if not (p[0] & used):
            return (p[0], p[1], p)


def _random_fleet(rng):
    fleet, used = [], 0
    for _, length in SHIPS:
        item = _legal_random(length, used, rng)
        used |= item[0]
        fleet.append(item)
    return fleet


def _noisy(weights, rng, noise):
    if not noise:
        return weights
    return [[w * (1 - noise + 2 * noise * rng.random()) for w in row] for row in weights]


def _to_desc(fleet):
    return [(name, length, p[2], p[3], p[4])
            for (name, length), (_, _, p) in zip(SHIPS, fleet)]


def _others(fleet, i):
    m = 0
    for j, (mask, _, _) in enumerate(fleet):
        if j != i:
            m |= mask
    return m


# --------------------------------------------------------------------------
def random_hide(weights, rng=None, noise=0.15, touch_penalty=0.8, **_):
    rng = rng or random.Random()
    w = _noisy(weights, rng, noise)
    fleet = _random_fleet(rng)
    s = _score(fleet, w, touch_penalty)
    return _to_desc(fleet), {"method": "random", "best": s, "start_avg": s, "trace": [s] * BUDGET,
                             "text": f"AI hid its fleet at random (score {s:.1f})."}


def hill_climb_fleet(weights, rng=None, restarts=10, steps=250, touch_penalty=0.8, noise=0.15, **_):
    rng = rng or random.Random()
    w = _noisy(weights, rng, noise)
    best, best_score, finals, starts, trace = None, float("inf"), [], [], []
    for _ in range(restarts):
        fleet = _random_fleet(rng)
        cur = _score(fleet, w, touch_penalty)
        starts.append(cur)
        if cur < best_score:
            best, best_score = fleet, cur
        trace.append(best_score)
        for _ in range(steps):
            i = rng.randrange(len(SHIPS))
            p = rng.choice(placements(SHIPS[i][1]))
            if p[0] & _others(fleet, i):
                trace.append(best_score)
                continue
            cand = list(fleet)
            cand[i] = (p[0], p[1], p)
            s = _score(cand, w, touch_penalty)
            if s <= cur:                      # accept improvements AND sideways moves
                fleet, cur = cand, s
            if cur < best_score:
                best, best_score = fleet, cur
            trace.append(best_score)
        finals.append(cur)
    info = {"method": "hill", "restarts": restarts, "steps": steps, "best": best_score,
            "start_avg": sum(starts) / len(starts), "finals": finals, "trace": trace,
            "text": f"Hill climbing, {restarts} restarts: score {sum(starts) / len(starts):.1f} -> "
                    f"{best_score:.1f} (lower = harder to find)."}
    return _to_desc(best), info


def genetic_fleet(weights, rng=None, pop=50, gens=50, mutation=0.35, touch_penalty=0.8,
                  noise=0.15, **_):
    rng = rng or random.Random()
    w = _noisy(weights, rng, noise)
    trace, state = [], {"best": float("inf"), "fleet": None}

    def evaluate(f):
        s = _score(f, w, touch_penalty)
        if s < state["best"]:
            state["best"], state["fleet"] = s, f
        trace.append(state["best"])
        return s

    def tournament(scored):
        return min(rng.sample(scored, 3), key=lambda x: x[0])[1]

    def crossover(a, b):
        child, used = [], 0
        for i, (_, length) in enumerate(SHIPS):
            first, second = (a[i], b[i]) if rng.random() < 0.5 else (b[i], a[i])
            if not (first[0] & used):
                pick = first
            elif not (second[0] & used):
                pick = second
            else:
                pick = _legal_random(length, used, rng)          # repair an overlap
            used |= pick[0]
            child.append(pick)
        return child

    def mutate(f):
        i = rng.randrange(len(SHIPS))
        f = list(f)
        f[i] = _legal_random(SHIPS[i][1], _others(f, i), rng)
        return f

    scored = [(evaluate(f), f) for f in (_random_fleet(rng) for _ in range(pop))]
    start = sum(s for s, _ in scored) / pop
    for _ in range(gens - 1):
        scored.sort(key=lambda x: x[0])
        nxt = scored[:2]                                           # elitism
        while len(nxt) < pop:
            child = crossover(tournament(scored), tournament(scored))
            if rng.random() < mutation:
                child = mutate(child)
            nxt.append((evaluate(child), child))
        scored = nxt
    while len(trace) < BUDGET:
        trace.append(state["best"])
    info = {"method": "genetic", "best": state["best"], "start_avg": start, "trace": trace,
            "pop": pop, "gens": gens,
            "text": f"Genetic algorithm, {pop} fleets x {gens} generations: population average "
                    f"{start:.1f} -> best {state['best']:.1f}."}
    return _to_desc(state["fleet"]), info


HIDERS = {"random": random_hide, "hill": hill_climb_fleet, "genetic": genetic_fleet}
