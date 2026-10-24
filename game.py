"""
game.py - game flow for YOU vs the AI, kept separate from drawing so it can be tested headlessly.

Phases: "place" (you position your fleet) -> "battle" -> "over".
The opponent is configurable: cfg = {"brain": ..., "learn": {...}, "hider": ...}
(see ai_player.py for brains and learners, hiding.py for hiding methods).
"""
import random

from ai_player import AIPlayer
from board import Board, SHIPS, cell_name, placements
from hiding import hide


def default_cfg():
    return {"brain": "csp", "learn": set(), "hider": "hill"}


class Game:
    def __init__(self, profile, rng=None, ai_samples=200, cfg=None):
        self.profile = profile
        self.rng = rng or random.Random()
        self.ai_samples = ai_samples
        self.cfg = cfg or default_cfg()
        self.phase = "place"
        self.turn = "player"
        self.winner = None
        self.player = Board()
        self.enemy = None
        self.ai = None
        self.pending = list(SHIPS)
        self.horizontal = True
        self.player_shots = []
        self.player_msg = ""
        self.ai_msg = ""
        self.last_ai = None          # last AI decision (explanation, stats)
        self.placement_info = None   # info from the method the AI used to hide its fleet

    # ---- placing your fleet ---------------------------------------------
    def can_place(self, r, c, horizontal=None):
        if not self.pending:
            return False
        h = self.horizontal if horizontal is None else horizontal
        return self.player.can_place(self.pending[0][1], r, c, h)

    def place(self, r, c):
        if not self.can_place(r, c):
            return False
        name, length = self.pending.pop(0)
        self.player.place(name, length, r, c, self.horizontal)
        return True

    def undo(self):
        ship = self.player.remove_last()
        if ship:
            self.pending.insert(0, (ship.name, ship.length))

    def auto_place(self):
        while self.pending:
            _, length = self.pending[0]
            _, _, r, c, h = self.rng.choice(placements(length))
            self.horizontal = h
            self.place(r, c)

    # ---- battle ----------------------------------------------------------
    def start_battle(self):
        fleet, self.placement_info = hide(self.cfg["hider"], self.profile.shot_weights(), self.rng)
        self.enemy = Board()
        self.enemy.place_fleet(fleet)
        self.ai = AIPlayer(self.cfg["brain"], self.rng, self.profile, self.ai_samples,
                           learn=self.cfg["learn"])
        self.phase = "battle"
        self.turn = "player"
        self.player_msg = "Your move: click a cell in the enemy waters."

    def player_fire(self, r, c):
        if self.phase != "battle" or self.turn != "player" or (r, c) in self.enemy.shots:
            return False
        result, ship = self.enemy.fire(r, c)
        self.player_shots.append((r, c))
        name = cell_name(r, c)
        if result == "miss":
            self.player_msg = f"You fired at {name}: miss."
        elif ship.sunk:
            self.player_msg = f"You fired at {name}: HIT - you sank the {ship.name}!"
        else:
            self.player_msg = f"You fired at {name}: HIT!"
        if self.enemy.all_sunk():
            self.finish("player")
        else:
            self.turn = "ai"
        return True

    def ai_move(self):
        if self.phase != "battle" or self.turn != "ai":
            return
        d = self.ai.choose()
        r, c = d["cell"]
        result, ship = self.player.fire(r, c)
        sunk = ship if (ship is not None and ship.sunk) else None
        self.ai.observe(r, c, result, sunk)
        self.last_ai = d
        name = cell_name(r, c)
        if result == "miss":
            self.ai_msg = f"AI fired at {name}: miss."
        elif sunk:
            self.ai_msg = f"AI fired at {name}: HIT - it sank your {ship.name}!"
        else:
            self.ai_msg = f"AI fired at {name}: HIT!"
        if self.player.all_sunk():
            self.finish("ai")
        else:
            self.turn = "player"

    def finish(self, winner):
        self.phase = "over"
        self.winner = winner
        self.profile.add_game(self.player.fleet_description(), self.player_shots, winner)
