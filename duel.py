"""
duel.py - the hiding duel: minimax with alpha-beta pruning.

A two-ply zero-sum game:

    MAX node (the hider)  : chooses ONE candidate fleet
    MIN node (the seeker) : answers with the seeker algorithm that finds it fastest
    leaf value            : average number of shots the seeker needs (the hider wants it high)

    value = max over fleets ( min over seekers ( shots ) )

Every leaf is a batch of simulated games, so each leaf is expensive. Alpha-beta pruning skips
leaves: once a fleet's running minimum drops to the best value found so far (alpha), the
remaining seekers cannot rescue that fleet, so they are never simulated. Seekers are ordered
strongest-first so the minimum drops quickly. Minimax and alpha-beta return the same value.
"""
import random

from ai_player import AIPlayer
from board import Board
from placement import HIDERS

SEEKERS = ["density", "hunt", "random"]        # strongest first -> better pruning


def _play(fleet, seeker, seed):
    board = Board()
    board.place_fleet(fleet)
    ai = AIPlayer(seeker, random.Random(seed))
    shots = 0
    while not board.all_sunk():
        r, c = ai.choose()["cell"]
        res, ship = board.fire(r, c)
        ai.observe(r, c, res, ship if (ship is not None and ship.sunk) else None)
        shots += 1
    return shots


class _Leaves:
    """Leaf evaluator with a counter (how many leaves were actually simulated)."""

    def __init__(self, fleets, seekers, games):
        self.fleets, self.seekers, self.games = fleets, seekers, games
        self.count = 0
        self.cache = {}

    def __call__(self, fi, si):
        if (fi, si) not in self.cache:
            self.count += 1
            seeds = [7919 * fi + 104729 * si + g for g in range(self.games)]
            vals = [_play(self.fleets[fi], self.seekers[si], s) for s in seeds]
            self.cache[(fi, si)] = sum(vals) / len(vals)
        return self.cache[(fi, si)]


def minimax(fleets, seekers, games=2):
    leaf = _Leaves(fleets, seekers, games)
    best, arg, matrix = float("-inf"), 0, []
    for fi in range(len(fleets)):
        row = [leaf(fi, si) for si in range(len(seekers))]
        matrix.append(row)
        if min(row) > best:
            best, arg = min(row), fi
    return {"value": best, "chosen": arg, "matrix": matrix, "leaves": leaf.count}


def alphabeta(fleets, seekers, games=2):
    leaf = _Leaves(fleets, seekers, games)
    alpha, arg, matrix = float("-inf"), 0, []
    for fi in range(len(fleets)):
        v, pruned, row = float("inf"), False, [None] * len(seekers)
        for si in range(len(seekers)):
            row[si] = leaf(fi, si)
            v = min(v, row[si])
            if v <= alpha:                 # MIN already holds this fleet at or below alpha
                pruned = True
                break
        matrix.append(row)
        if not pruned and v > alpha:
            alpha, arg = v, fi
    return {"value": alpha, "chosen": arg, "matrix": matrix, "leaves": leaf.count}


def candidates(weights, rng, quick=True):
    """A small pool of fleets from different optimisers, plus one random fleet."""
    pool = [("hill", HIDERS["hill"](weights, rng, restarts=4, steps=100)),
            ("genetic", HIDERS["genetic"](weights, rng, pop=24, gens=20)),
            ("random", HIDERS["random"](weights, rng))]
    return [(label, fleet) for label, (fleet, _) in pool]


def choose_fleet(weights, rng, games=2, compare=False):
    """Pick the fleet that minimax says is hardest to find. Returns (fleet, info)."""
    cands = candidates(weights, rng)
    fleets = [f for _, f in cands]
    ab = alphabeta(fleets, SEEKERS, games)
    info = {"method": "minimax", "labels": [l for l, _ in cands], "seekers": list(SEEKERS),
            "matrix": ab["matrix"], "chosen": ab["chosen"], "value": ab["value"],
            "leaves_ab": ab["leaves"], "leaves_mm": len(fleets) * len(SEEKERS), "best": ab["value"]}
    if compare:
        mm = minimax(fleets, SEEKERS, games)
        info["minimax_value"], info["leaves_mm"] = mm["value"], mm["leaves"]
    info["text"] = (f"Minimax duel: picked the '{info['labels'][ab['chosen']]}' fleet - the best seeker "
                    f"needs {ab['value']:.0f} shots. Alpha-beta simulated {ab['leaves']} of "
                    f"{len(fleets) * len(SEEKERS)} leaves.")
    return fleets[ab["chosen"]], info
