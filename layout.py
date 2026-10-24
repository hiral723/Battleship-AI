"""
layout.py - bitmask helpers and layout features.

A fleet layout is a tuple of ship masks (one 100-bit int per ship). From it we compute
three features that K-means uses to describe a player's *style*:

    edge   : fraction of ship cells on the outer ring of the board
    touch  : fraction of ships that touch another ship
    horiz  : fraction of ships that are horizontal

`style_weight` is a likelihood ratio: how much more probable a layout is under your style than
under a player with no habits. The CSP uses it to count layouts that match your style MORE than
layouts that do not (weighted counting). With no habits the ratio is 1, so nothing changes.
"""
import math

from board import SIZE, placements

FULL = (1 << (SIZE * SIZE)) - 1
COL0 = sum(1 << (r * SIZE) for r in range(SIZE))
COLN = sum(1 << (r * SIZE + SIZE - 1) for r in range(SIZE))
EDGE = 0
for _r in range(SIZE):
    for _c in range(SIZE):
        if _r in (0, SIZE - 1) or _c in (0, SIZE - 1):
            EDGE |= 1 << (_r * SIZE + _c)

_ORIENT = {}
for _length in (2, 3, 4, 5):
    for _mask, _cells, _r, _c, _h in placements(_length):
        _ORIENT[_mask] = _h


def halo(m):
    """Cells orthogonally adjacent to the cells of mask m."""
    return (((m << 1) & ~COL0) | ((m >> 1) & ~COLN) | (m << SIZE) | (m >> SIZE)) & FULL


def is_horizontal(mask):
    return _ORIENT[mask]


def layout_features(masks):
    n = len(masks)
    total = sum(m.bit_count() for m in masks)
    edge = sum((m & EDGE).bit_count() for m in masks) / total
    touching = 0
    for i, m in enumerate(masks):
        h = halo(m)
        if any(h & o for j, o in enumerate(masks) if j != i):
            touching += 1
    horiz = sum(1 for m in masks if _ORIENT[m]) / n
    return [edge, touching / n, horiz]


_RANDOM_STATS = None


def random_stats():
    """(mean, variance) of the three layout features over uniformly random fleets.

    This is what a player with NO habits looks like, so it is the yardstick for "does this
    player's style carry real information?". Computed once from 400 fixed-seed fleets."""
    global _RANDOM_STATS
    if _RANDOM_STATS is None:
        import random
        from board import Board
        rng = random.Random(2024)
        feats = [layout_features(tuple(mask_of(s.cells) for s in Board.random(rng).ships))
                 for _ in range(400)]
        n = len(feats)
        mean = [sum(f[k] for f in feats) / n for k in range(3)]
        var = [max(1e-4, sum((f[k] - mean[k]) ** 2 for f in feats) / n) for k in range(3)]
        _RANDOM_STATS = (mean, var)
    return _RANDOM_STATS


def style_log_weight(masks, centre, var):
    """log of the importance weight of a candidate layout (see style_weight), clamped."""
    f = layout_features(masks)
    mu, var_r = random_stats()
    lr = 0.0
    for k in range(3):
        lr += (-(f[k] - centre[k]) ** 2 / (2 * var[k]) + (f[k] - mu[k]) ** 2 / (2 * var_r[k])
               - 0.5 * math.log(var[k] / var_r[k]))
    return max(-12.0, min(4.0, lr))


def style_weight(masks, centre, var, power=1.0):
    """Importance weight of a candidate layout = P(layout | this player's style) / P(layout | a
    player with no habits), each modelled as an independent Gaussian per feature.

    centre, var : the style's mean and variance per feature (see PlayerProfile.style_model).
    power       : tempering exponent in [0, 1]. The learner lowers it when the evidence for a
                  style is weak, so a player without habits cannot distort the AI's beliefs.
    The ratio is exactly 1 for a style that looks like a random player; it only grows where the
    player really is predictable (tiny variance on a feature)."""
    return math.exp(power * style_log_weight(masks, centre, var))


def mask_of(cells):
    m = 0
    for r, c in cells:
        m |= 1 << (r * SIZE + c)
    return m
