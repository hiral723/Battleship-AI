"""
hiding.py - one entry point for every way the AI can hide its fleet.

    methods: random, hill, genetic  (placement.py)   and   minimax  (duel.py)
"""
import duel
from placement import HIDERS

METHODS = ["random", "hill", "genetic", "minimax"]
HIDE_TEXT = {
    "random": "A random legal fleet.",
    "hill": "Hill climbing with sideways moves and restarts.",
    "genetic": "Genetic algorithm: evolves a population of fleets.",
    "minimax": "Minimax + alpha-beta duel against several seeker algorithms.",
}


def hide(method, weights, rng):
    """Returns (fleet, info). fleet = list of (name, length, row, col, horizontal)."""
    if method == "minimax":
        return duel.choose_fleet(weights, rng)
    return HIDERS[method](weights, rng)
