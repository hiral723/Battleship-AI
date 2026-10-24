"""
habits.py - scripted "habit players" used to test whether the AI really learns.

    edge       : hides ships along the board edges
    clump      : hides ships touching each other
    horizontal : lays every ship horizontally
    repeat     : uses the same fleet every single game
"""
import random

from board import Board, SHIPS, placements
from layout import EDGE, halo

KINDS = ["edge", "clump", "horizontal", "repeat"]
REPEAT_FLEET = Board.random(random.Random(12345)).fleet_description()


def _pick(length, used, rng, pred):
    for _ in range(500):
        p = rng.choice(placements(length))
        if not (p[0] & used) and pred(p, used):
            return p
    while True:                                  # fall back to any legal placement
        p = rng.choice(placements(length))
        if not (p[0] & used):
            return p


def habit_fleet(kind, rng):
    """A fleet as a list of (name, length, row, col, horizontal)."""
    if kind == "repeat":
        return list(REPEAT_FLEET)
    preds = {
        "edge": lambda p, used: (p[0] & EDGE) == p[0],
        "clump": lambda p, used: used == 0 or bool(halo(p[0]) & used),
        "horizontal": lambda p, used: p[4],
    }
    pred, used, out = preds[kind], 0, []
    for name, length in SHIPS:
        p = _pick(length, used, rng, pred)
        used |= p[0]
        out.append((name, length, p[2], p[3], p[4]))
    return out
