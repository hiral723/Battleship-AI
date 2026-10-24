"""
probability.py - turns CSP solutions into a probability heatmap.

P(cell has a ship) = (weight of consistent fleet layouts that use the cell)
                     / (weight of all consistent layouts)

Every layout normally weighs 1 (plain counting). Two learning hooks change the weights:

    log_weight_fn(masks) -> float   K-means style prior: returns the LOG weight of a layout, so
                                    layouts that look like YOUR style count more. The weights are
                                    tempered (raised to a power tau <= 1) just enough to keep the
                                    effective sample size >= MIN_ESS, so a very sharp style cannot
                                    make the estimate hang on two or three layouts.
    replay                          layouts you have used in several earlier games are injected as
                                    extra candidates (a case-based "replay detector"). They drop out
                                    automatically the moment a shot contradicts them.

If the CSP can list *all* layouts the result is exact; otherwise it is estimated from
random samples of layouts (and, with a style prior, more are drawn when too few of them count).
"""
import math
import time

from board import SIZE
from csp_solver import FleetCSP

MIN_ESS = 25          # minimum effective sample size of the style-weighted layouts


def _weights(logw, tau):
    m = max(tau * l for l in logw)
    return [math.exp(tau * l - m) for l in logw]


def _ess(ws):
    """Effective sample size (Kish): how many equally-weighted layouts the weights are worth."""
    s = sum(ws)
    return s * s / sum(w * w for w in ws)


def _temper(logw, target):
    """Largest tau in [0, 1] for which the weights exp(tau * logw) keep ESS >= target."""
    if _ess(_weights(logw, 1.0)) >= target:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if _ess(_weights(logw, mid)) >= target:
            lo = mid
        else:
            hi = mid
    return lo


def estimate(kb, rng, n_samples=200, exact_cap=1500, exact_nodes=30000,
             log_weight_fn=None, replay=None):
    """Returns a dict with prob (100 floats), n, exact, nodes, ms, ess, tau, injected - or None
    if no consistent layout could be found."""
    t0 = time.perf_counter()
    csp = FleetCSP(kb, rng)
    if not csp.vars:
        return None
    sols = None
    # Exact enumeration only pays off when the board is constrained.
    if csp.hit_mask or len(csp.vars) <= 2:
        sols = csp.enumerate_exact(exact_cap, exact_nodes)
    exact = sols is not None
    if not exact:
        sols = csp.sample(n_samples)
    if not sols:
        return None

    tau = 1.0
    if log_weight_fn is not None:
        logw = [log_weight_fn(masks) for _, masks in sols]
        if not exact:
            # A sharp style can leave only a handful of layouts that really count: draw more
            # candidate layouts first (at most 3 extra rounds), then temper whatever is left.
            for _ in range(3):
                if _ess(_weights(logw, 1.0)) >= MIN_ESS:
                    break
                more = csp.sample(n_samples)
                if not more:
                    break
                sols = sols + more
                logw += [log_weight_fn(masks) for _, masks in more]
        tau = _temper(logw, min(MIN_ESS, 0.5 * len(logw)))
        ws = _weights(logw, tau)
    else:
        ws = [1.0] * len(sols)
    entries = [(occ, w) for (occ, _), w in zip(sols, ws)]
    ess = _ess(ws)
    wsum = sum(ws)

    injected = 0
    if replay and replay.get("m", 0) > 0:
        found = []
        for fleet, share in replay["fleets"]:
            hit = csp.consistent(fleet)
            if hit:
                found.append((hit[0], share))
        if found:
            mass = replay["m"] / (1 - replay["m"]) * wsum
            for occ, share in found:
                entries.append((occ, mass * share))
            injected = len(found)

    counts = [0.0] * (SIZE * SIZE)
    total = 0.0
    for occ, w in entries:
        total += w
        o = occ
        while o:
            low = o & -o
            counts[low.bit_length() - 1] += w
            o ^= low
    return {"prob": [c / total for c in counts], "n": len(sols), "exact": exact,
            "nodes": csp.nodes, "ms": (time.perf_counter() - t0) * 1000,
            "ess": ess, "tau": tau, "injected": injected}
