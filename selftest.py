"""
selftest.py - correctness checks for every module. No pygame needed (about 20 seconds).

    python selftest.py
"""
import os
import random
import tempfile

from ai_player import AIPlayer, BRAINS, LEARNERS, _posterior
from board import Board, SHIPS, TOTAL_SHIP_CELLS, idx, placements
from game import Game
from knowledge_base import KnowledgeBase
from placement import hill_climb_fleet
from player_model import PlayerProfile, DecisionTree, kmeans, ARCHETYPES
from probability import estimate
import resolution

passed = 0


def check(name, cond):
    global passed
    if not cond:
        raise SystemExit(f"FAIL: {name}")
    passed += 1
    print(f"  ok  {name}")


rng = random.Random(3)

print("board")
check("120 placements for a length-5 ship", len(placements(5)) == 120)
check("180 placements for a length-2 ship", len(placements(2)) == 180)
b = Board.random(rng)
check("random fleet has 17 distinct cells", len(b.occupied) == TOTAL_SHIP_CELLS == 17)
d = Board()
d.place("Destroyer", 2, 0, 0, True)
check("hit then sink", d.fire(0, 0)[0] == "hit" and d.fire(0, 1)[1].sunk and d.all_sunk())
check("repeat shot detected", d.fire(0, 0)[0] == "repeat")

print("logic")
ok, steps = resolution.entails([frozenset([("A", True), ("B", True)]), frozenset([("A", False)])], ("B", True))
check("resolution proves B from (A or B), ~A", ok and steps)
ok, _ = resolution.entails([frozenset([("A", True), ("B", True)])], ("B", True))
check("resolution does not prove B from (A or B) alone", not ok)
kb = KnowledgeBase()
kb.tell(0, 1, "miss"); kb.tell(1, 0, "miss")
kb.forward_chain()
check("isolated corner cell is Dead (R4)", ("Dead", (0, 0)) in kb.facts)
kb = KnowledgeBase()
kb.tell(0, 0, "hit"); kb.tell(0, 1, "miss")
kb.forward_chain()
check("corner hit + miss => other neighbour Forced (R5)", ("Forced", (1, 0)) in kb.facts and (1, 0) in kb.proofs)

print("CSP + probability")
kb = KnowledgeBase()
e = estimate(kb, rng, 150)
check("heatmap sums to 17 ship cells (empty board)", abs(sum(e["prob"]) - 17) < 1e-9)
kb.tell(4, 4, "hit")
e = estimate(kb, rng, 150)
check("heatmap sums to 17 with a hit", abs(sum(e["prob"]) - 17) < 1e-9)
check("hit cell has probability 1", e["prob"][idx(4, 4)] == 1.0)
check("neighbour of a hit is hotter than a far corner", e["prob"][idx(4, 5)] > 3 * e["prob"][0])
kb = KnowledgeBase()
for col in range(4, 9):                      # five hits in a row, nothing sunk
    kb.tell(4, col, "hit")
e = estimate(kb, rng, 200)
check("an unsunk ship cannot lie fully on hit cells (constraint 3)",
      e["prob"][idx(4, 3)] > 0.4 and e["prob"][idx(4, 9)] > 0.4)
kb = KnowledgeBase()
kb.remaining = [("Destroyer", 2)]
kb.tell(5, 5, "hit")
for cell in [(5, 4), (5, 6), (4, 5)]:
    kb.tell(*cell, "miss")
e = estimate(kb, rng)
check("exact enumeration finds the only layout", e["exact"] and e["n"] == 1 and e["prob"][idx(6, 5)] == 1.0)

print("AI agent")
for strategy in BRAINS:
    board = Board.random(rng)
    ai = AIPlayer(strategy, rng, samples=60)
    fired = set()
    while not board.all_sunk():
        cell = ai.choose()["cell"]
        assert cell not in fired, f"{strategy} repeated {cell}"
        fired.add(cell)
        res, ship = board.fire(*cell)
        ai.observe(*cell, res, ship if (ship is not None and ship.sunk) else None)
    check(f"{strategy} agent sinks the fleet without repeating a shot ({len(fired)} shots)", len(fired) <= 100)

print("learning and placement search")
rows = [{"x": "a"}] * 20 + [{"x": "b"}] * 20
tree = DecisionTree(max_depth=2).fit(rows, [1] * 20 + [0] * 20)
check("decision tree separates two classes", tree.predict({"x": "a"}) > 0.9 > 0.1 > tree.predict({"x": "b"}))
cents = kmeans([[0, 0], [0.1, 0], [1, 1], [0.9, 1]], [[0, 0], [1, 1]])
check("k-means finds two clusters", cents[0][0] < 0.2 and cents[1][0] > 0.8)
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "h.json")
    prof = PlayerProfile(path)
    check("no bias before any games", prof.cell_bias(0, 0) == 1.0)
    for _ in range(3):
        fleet = Board.random(rng).fleet_description()
        prof.add_game(fleet, [(rng.randrange(10), rng.randrange(10)) for _ in range(30)], "ai")
    check("history is saved to disk", os.path.exists(path))
    prof2 = PlayerProfile(path)
    check("history reloads", len(prof2.games) == 3 and prof2.style in ARCHETYPES)
    check("bias stays in range", all(0.4 <= prof2.cell_bias(r, c) <= 2.5 for r in range(10) for c in range(10)))
    w = prof2.shot_weights()
    fleet, info = hill_climb_fleet(w, rng)
    hidden = Board(); hidden.place_fleet(fleet)
    check("hill climbing returns a legal 17-cell fleet", len(hidden.occupied) == 17)
    check("best restart beats the average random start", info["best"] <= info["start_avg"])
    g = Game(prof2, rng, ai_samples=40)
    g.auto_place(); g.start_battle()
    while g.phase == "battle":
        if g.turn == "player":
            cell = rng.choice([(r, c) for r in range(10) for c in range(10) if (r, c) not in g.enemy.shots])
            g.player_fire(*cell)
        else:
            g.ai_move()
    check("full game runs to a winner and is recorded", g.winner in ("player", "ai") and len(prof2.games) == 4)

print("learning fixes")
from habits import habit_fleet
from layout import layout_features, mask_of
import probability

check("five brains and three learners are registered",
      BRAINS == ["random", "hunt", "logic", "density", "csp"] and LEARNERS == ["tree", "kmeans", "replay"])
check("odds update keeps a certain cell certain and an impossible one impossible",
      _posterior(1.0, 0.4) == 1.0 and _posterior(0.0, 2.5) == 0.0)
check("learning can never outrank a certain cell with a long shot",
      _posterior(1.0, 0.4) > _posterior(0.49, 2.5))
check("odds above 1 raise and below 1 lower a middling probability",
      _posterior(0.3, 2.0) > 0.3 > _posterior(0.3, 0.5))

hrng = random.Random(8)
rand_prof, edge_prof = PlayerProfile(None), PlayerProfile(None)
for _ in range(6):
    rand_prof.add_game(Board.random(hrng).fleet_description(), [], "ai")
    edge_prof.add_game(habit_fleet("edge", hrng), [], "ai")
check("a player with no habits is labelled 'No clear habit'",
      rand_prof.style_name == "No clear habit" and rand_prof.style_evidence() < 1.0)
check("an edge hugger is recognised with strong evidence",
      edge_prof.style_name == "Edge hugger" and edge_prof.style_evidence() > 2.0)


def masks_of(board):
    return tuple(mask_of(s.cells) for s in board.ships)


rand_fn, edge_fn = rand_prof.kmeans_weight_fn(), edge_prof.kmeans_weight_fn()
rand_boards = [Board.random(hrng) for _ in range(40)]
edge_board = Board(); edge_board.place_fleet(habit_fleet("edge", hrng))
check("style weights are nearly flat for a player with no habits",
      max(abs(rand_fn(masks_of(b))) for b in rand_boards) < 1.5)
rand_logw = sorted(edge_fn(masks_of(b)) for b in rand_boards)
check("style weights favour edge layouts for an edge hugger",
      edge_fn(masks_of(edge_board)) > rand_logw[len(rand_logw) // 2] + 2.0)
sharp = [0.0] * 50 + [12.0] * 2
tau = probability._temper(sharp, 25)
check("tempering restores a healthy effective sample size",
      tau < 1.0 and probability._ess(probability._weights(sharp, tau)) >= 24)
check("flat weights need no tempering", probability._temper([0.0] * 30, 25) == 1.0)
e = probability.estimate(KnowledgeBase(), random.Random(1), 100, log_weight_fn=edge_fn)
check("a sharp style prior still leaves a reliable estimate (ESS >= 24)", e["ess"] >= 24 and abs(sum(e["prob"]) - 17) < 1e-6)
ai = AIPlayer("csp", random.Random(2), edge_prof, 80, learn=set(LEARNERS))
d = ai.choose()
check("the decision records its steps and a candidate table for the UI",
      len(d["steps"]) == 3 and d["table"] and d["table"][0][0] == d["cell"])

print("first-order logic")
import fol
check("unification binds variables", fol.unify(fol.parse_atom("P(?x,B1)"), fol.parse_atom("P(A1,?y)"), {}) ==
      {"?x": "A1", "?y": "B1"})
check("unification fails on a clash", fol.unify(fol.parse_atom("P(A1)"), fol.parse_atom("P(B1)"), {}) is None)
kb = KnowledgeBase()
kb.tell(0, 0, "hit"); kb.tell(0, 1, "miss")
eng = fol.engine_for(kb)
res = eng.ask("ForcedShip(?c)")
check("backward chaining proves a forced ship (corner hit + miss)",
      res["answers"] and res["answers"][0]["bindings"]["?c"] == "A2" and res["answers"][0]["proof"])
check("backward chaining proves NoShip for a missed cell", eng.ask("NoShip(B1)")["answers"])
check("backward chaining cannot prove a ship on an empty board",
      not fol.engine_for(KnowledgeBase()).ask("Ship(E5)")["answers"])
check("forward chaining derives facts", len(eng.forward()) > 3)

print("fleet hiding and the minimax duel")
import duel
from board import SIZE
from hiding import hide, METHODS


def rugged_weights(seed):
    r = random.Random(seed)
    w = [[0.6] * SIZE for _ in range(SIZE)]
    for _ in range(9):
        cr, cc, amp = r.randrange(SIZE), r.randrange(SIZE), r.uniform(1.5, 3.5)
        for a in range(SIZE):
            for b in range(SIZE):
                w[a][b] += amp * max(0.0, 1 - (abs(a - cr) + abs(b - cc)) / 3.0)
    return w


w = rugged_weights(1)
scores = {}
for m in METHODS:
    fleet, info = hide(m, w, random.Random(5))
    b = Board(); b.place_fleet(fleet)
    check(f"hider '{m}' returns a legal 17-cell fleet", len(b.occupied) == 17)
    scores[m] = info["best"]
check("hill and genetic both beat a random fleet by a wide margin",
      all(scores[m] < scores["random"] - 8 for m in ("hill", "genetic")))
fleets = [f for _, f in duel.candidates(w, random.Random(3))]
mm = duel.minimax(fleets, duel.SEEKERS, 3)
ab = duel.alphabeta(fleets, duel.SEEKERS, 3)
check("alpha-beta returns the same value as minimax", abs(ab["value"] - mm["value"]) < 1e-9)
check("alpha-beta simulates fewer leaves than minimax", ab["leaves"] < mm["leaves"])

print("AI vs AI arena")
import arena
cfg = arena.parse_cfg("csp:tree+kmeans+replay@genetic")
check("arena config strings parse", cfg == {"brain": "csp", "learn": {"tree", "kmeans", "replay"}, "hider": "genetic"})
for bad in ("nonsense", "csp:magic", "csp@teleport"):
    try:
        arena.parse_cfg(bad)
        check(f"bad config '{bad}' is rejected", False)
    except ValueError:
        check(f"bad config '{bad}' is rejected", True)
summary = arena.run_series(arena.parse_cfg("density@hill"), arena.parse_cfg("hunt@random"), games=2, seed=1, samples=40)
check("an AI vs AI series plays to a winner every game",
      summary["wins_a"] + summary["wins_b"] == 2 and all(r["winner"] in ("A", "B") for r in summary["records"]))

print(f"\nAll {passed} checks passed.")
