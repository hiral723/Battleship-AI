"""
benchmark.py - how many shots does each shooting brain need to sink the same fleets?

    python benchmark.py                       # about 1 minute
    python benchmark.py --games 100 --csp-games 30
    python benchmark.py --chart               # also saves benchmark.png (needs matplotlib)

Lower is better (100 is the worst possible). Every brain faces exactly the same fleets, so the
comparison is paired. The csp brain runs WITHOUT learning here, because random fleets have no
habits to learn.
"""
import argparse
import random
import statistics
import time

from ai_player import AIPlayer, BRAINS
from board import Board


def play(brain, fleet, seed, samples):
    board = Board()
    board.place_fleet(fleet)
    ai = AIPlayer(brain, random.Random(seed), samples=samples)
    shots, think = 0, 0.0
    while not board.all_sunk():
        t = time.perf_counter()
        r, c = ai.choose()["cell"]
        think += time.perf_counter() - t
        res, ship = board.fire(r, c)
        ai.observe(r, c, res, ship if (ship is not None and ship.sunk) else None)
        shots += 1
    return shots, think / shots * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="fleets for the fast brains")
    ap.add_argument("--csp-games", type=int, default=60, help="fleets for the slower csp brain")
    ap.add_argument("--samples", type=int, default=120, help="layouts the csp brain samples per move")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--chart", action="store_true")
    args = ap.parse_args()

    fleets = [Board.random(random.Random(args.seed + i)).fleet_description() for i in range(args.games)]
    results = {}
    print(f"{'brain':<9}{'games':>6}{'mean':>7}{'+-':>6}{'median':>8}{'best':>6}{'worst':>7}{'ms/move':>9}")
    for brain in BRAINS:
        n = min(args.csp_games, args.games) if brain == "csp" else args.games
        shots, ms = [], []
        for i in range(n):
            s, m = play(brain, fleets[i], i, args.samples)
            shots.append(s)
            ms.append(m)
        results[brain] = shots
        se = statistics.pstdev(shots) / n ** 0.5
        print(f"{brain:<9}{n:>6}{statistics.mean(shots):>7.1f}{se:>6.1f}{statistics.median(shots):>8.1f}"
              f"{min(shots):>6}{max(shots):>7}{statistics.mean(ms):>9.1f}")

    def stats(shots):
        return statistics.mean(shots), statistics.pstdev(shots) / len(shots) ** 0.5

    ranked = sorted(results.items(), key=lambda kv: statistics.mean(kv[1]))
    best_mean, best_se = stats(ranked[0][1])
    print("\nRanking (fewest shots first):")
    for rank, (brain, shots) in enumerate(ranked, 1):
        mean, se = stats(shots)
        tie = rank > 1 and abs(mean - best_mean) < 2 * (se ** 2 + best_se ** 2) ** 0.5
        print(f"  {rank}. {brain:<8}{mean:6.1f} shots" + ("   (statistically tied with #1)" if tie else ""))
    print("\n'+-' is the standard error of the mean. Means closer than about two standard errors "
          "cannot be told apart with this many games.")

    if args.chart:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed - run: pip install matplotlib")
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.boxplot([results[b] for b in BRAINS], tick_labels=BRAINS)
        ax.set_ylabel("Shots to sink the fleet (lower is better)")
        ax.set_title("Shooting brains on identical fleets")
        fig.tight_layout()
        fig.savefig("benchmark.png", dpi=150)
        print("saved benchmark.png")


if __name__ == "__main__":
    main()
