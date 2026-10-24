"""
arena.py - AI vs AI matches and series.

    python arena.py --a csp --b density --games 20
    python arena.py --a "csp:tree+kmeans+replay@genetic" --b "density@hill" --games 10

Config string:   brain[:learner+learner][@hider]
    brain   random | hunt | logic | density | csp
    learner tree | kmeans | replay           (only the csp brain uses them)
    hider   random | hill | genetic | minimax

Each AI keeps an in-memory model of its OPPONENT across a series: after every game it learns
the opponent's fleet layout and shots, exactly like the game learns from a human player.
So a learning AI can improve during a series against an opponent with habits.
"""
import argparse
import random
import statistics

from ai_player import AIPlayer, BRAINS, LEARNERS
from board import Board
from hiding import hide, METHODS as HIDERS
from player_model import PlayerProfile


def default_cfg(brain="csp", learn=(), hider="hill"):
    return {"brain": brain, "learn": set(learn), "hider": hider}


def parse_cfg(text):
    hider = "hill"
    if "@" in text:
        text, hider = text.split("@", 1)
    learn = set()
    if ":" in text:
        text, ls = text.split(":", 1)
        learn = {x for x in ls.split("+") if x}
    if text not in BRAINS:
        raise ValueError(f"unknown brain {text!r} (choose from {', '.join(BRAINS)})")
    if hider not in HIDERS:
        raise ValueError(f"unknown hider {hider!r} (choose from {', '.join(HIDERS)})")
    bad = learn - set(LEARNERS)
    if bad:
        raise ValueError(f"unknown learner {sorted(bad)} (choose from {', '.join(LEARNERS)})")
    return default_cfg(text, learn, hider)


def cfg_label(cfg):
    s = cfg["brain"]
    if cfg["learn"] and cfg["brain"] == "csp":
        s += ":" + "+".join(x for x in LEARNERS if x in cfg["learn"])
    return f"{s}@{cfg['hider']}"


class Side:
    """One player: its own fleet, the AI that shoots at the OTHER fleet, and a model of the opponent."""

    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        self.profile = PlayerProfile(None)          # in-memory, never saved
        self.board = self.ai = self.hide_info = None
        self.shots = []

    def prepare(self, rng, samples):
        fleet, self.hide_info = hide(self.cfg["hider"], self.profile.shot_weights(), rng)
        self.board = Board()
        self.board.place_fleet(fleet)
        self.ai = AIPlayer(self.cfg["brain"], rng, self.profile, samples, learn=self.cfg["learn"])
        self.shots = []


class Match:
    """One game. step() fires a single shot, so a UI can animate it."""

    def __init__(self, side_a, side_b, rng, samples=120, first="A"):
        self.A, self.B = side_a, side_b
        self.A.prepare(rng, samples)
        self.B.prepare(rng, samples)
        self.turn, self.winner, self.finished = first, None, False

    def step(self):
        if self.winner:
            return None
        me, other = (self.A, self.B) if self.turn == "A" else (self.B, self.A)
        d = me.ai.choose()
        r, c = d["cell"]
        res, ship = other.board.fire(r, c)
        sunk = ship if (ship is not None and ship.sunk) else None
        me.ai.observe(r, c, res, sunk)
        me.shots.append((r, c))
        ev = {"side": self.turn, "cell": (r, c), "result": res, "sunk": sunk, "decision": d}
        if other.board.all_sunk():
            self.winner = self.turn
        else:
            self.turn = "B" if self.turn == "A" else "A"
        return ev

    def finish(self):
        """Let both AIs learn from the game (the opponent's fleet is revealed at the end)."""
        if self.finished:
            return
        self.finished = True
        self.A.profile.add_game(self.B.board.fleet_description(), self.B.shots,
                                "player" if self.winner == "B" else "ai")
        self.B.profile.add_game(self.A.board.fleet_description(), self.A.shots,
                                "player" if self.winner == "A" else "ai")

    def shots_by(self, side):
        return len((self.A if side == "A" else self.B).shots)


def run_series(cfg_a, cfg_b, games=10, seed=1, samples=120, progress=None):
    """Plays `games` matches (alternating who shoots first); returns a summary dict."""
    rng = random.Random(seed)
    a, b = Side("A", cfg_a), Side("B", cfg_b)
    records = []
    for g in range(games):
        m = Match(a, b, rng, samples, first="A" if g % 2 == 0 else "B")
        while m.winner is None:
            m.step()
        m.finish()
        records.append({"winner": m.winner, "shots_a": m.shots_by("A"), "shots_b": m.shots_by("B")})
        if progress:
            progress(g + 1, games)
    return summarise(records, cfg_a, cfg_b)


def summarise(records, cfg_a, cfg_b):
    def mean_when(side):
        v = [r["shots_" + side.lower()] for r in records if r["winner"] == side]
        return statistics.mean(v) if v else None
    return {"a": cfg_label(cfg_a), "b": cfg_label(cfg_b), "games": len(records),
            "wins_a": sum(r["winner"] == "A" for r in records),
            "wins_b": sum(r["winner"] == "B" for r in records),
            "mean_shots_a": statistics.mean(r["shots_a"] for r in records) if records else 0,
            "mean_shots_b": statistics.mean(r["shots_b"] for r in records) if records else 0,
            "mean_win_shots_a": mean_when("A"), "mean_win_shots_b": mean_when("B"), "records": records}


def print_summary(s):
    print(f"\nA = {s['a']}\nB = {s['b']}\n")
    print(f"games played:           {s['games']}")
    print(f"wins:                   A {s['wins_a']}   B {s['wins_b']}")
    wa = "-" if s["mean_win_shots_a"] is None else f"{s['mean_win_shots_a']:.1f}"
    wb = "-" if s["mean_win_shots_b"] is None else f"{s['mean_win_shots_b']:.1f}"
    print(f"avg shots needed in games it WON: A {wa}   B {wb}")
    print("shots per game (A | B):  " + "  ".join(f"{r['shots_a']}|{r['shots_b']}" for r in s["records"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI vs AI series")
    ap.add_argument("--a", default="csp:tree+kmeans+replay@genetic")
    ap.add_argument("--b", default="density@hill")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=120)
    args = ap.parse_args()
    print_summary(run_series(parse_cfg(args.a), parse_cfg(args.b), args.games, args.seed, args.samples,
                             progress=lambda i, n: print(f"  game {i}/{n}", end="\r", flush=True)))
