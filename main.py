"""
main.py - Pygame user interface for Battleship AI.

    python main.py

Menu:
    PLAY VS AI     you against the AI. Pick its shooting brain (random / hunt / logic / density / csp)
                   and how it hides its fleet (random / hill / genetic / minimax).
    AI VS AI       watch two AIs play a series and compare their results.

While a game is running:
    W  why the AI fired there (rules, proofs, first-order logic)
    /  ask the AI's logic engine a question, e.g.  ForcedShip(?c)   or   Ship(E5)
    H  toggle the heatmap (the AI's belief about where ships are)
"""
import argparse
import copy
import math
import os
import random
import time
from collections import Counter

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

import fol
from ai_player import BRAINS, BRAIN_TEXT
from arena import Match, Side, cfg_label, default_cfg as arena_cfg, summarise
from board import SIZE, SHIPS, COLS, cell_name, idx
from game import Game, default_cfg as play_cfg
from hiding import METHODS as HIDERS, HIDE_TEXT
from player_model import PlayerProfile

W, H = 1220, 790
CELL = 38
BOARD_PX = CELL * SIZE
LEFT = (60, 175)
RIGHT = (480, 175)
PANEL = pygame.Rect(900, 110, 300, 650)
OVR = pygame.Rect(40, 132, 840, 628)

BG = (11, 22, 38)
WATER = (21, 58, 96)
GRID = (44, 96, 138)
SHIP = (126, 140, 156)
SHIP_SUNK = (78, 54, 60)
HIT = (232, 70, 58)
MISS = (205, 224, 238)
TEXT = (228, 238, 248)
DIM = (140, 160, 182)
ACCENT = (255, 196, 70)
GOOD = (96, 214, 130)
BAD = (240, 96, 90)
BLUE_A = (96, 170, 255)
ORANGE_B = (255, 150, 70)
PURPLE = (190, 140, 240)
PANEL_BG = (18, 34, 56)
FAMILY_COLOR = {"uninformed": (96, 150, 230), "informed": ACCENT, "csp": GOOD, "local": PURPLE}


def make_font(size, bold=False):
    return pygame.font.SysFont("segoeui,arial,helvetica,dejavusans", size, bold=bold)


def wrap(text, font, width):
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if font.size(trial)[0] <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


class App:
    def __init__(self, history_path=None):
        pygame.display.init()          # no audio needed, so skip pygame.init() / the mixer
        pygame.font.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Battleship AI")
        self.clock = pygame.time.Clock()
        self.f_s, self.f_m = make_font(14), make_font(16)
        self.f_b, self.f_t = make_font(20, True), make_font(34, True)
        self.profile = PlayerProfile(history_path)
        self.cfg = {"play": play_cfg(),
                    "A": arena_cfg("csp", (), "genetic"),
                    "B": arena_cfg("density", (), "hill")}
        self.games_target = 10
        self.scene = "menu"
        self.hot = []
        self.overlay = None
        self.ovr_lines = []
        self.ask_text, self.ask_out = "", []
        self.show_heat, self.show_enemy = True, False
        # play
        self.game, self.ai_stage, self.ai_due = None, 0, 0.0
        # arena
        self.sa = self.sb = self.match = None
        self.records, self.series_done, self.arena_paused = [], False, False
        self.delay, self.view, self.view_lock, self.last_ev = 0.25, "A", False, None
        self.stage, self.next_t = 0, 0.0

    # ================================================================ helpers
    def txt(self, s, pos, font=None, color=TEXT):
        self.screen.blit((font or self.f_m).render(s, True, color), pos)

    def para(self, s, x, y, width, font=None, color=TEXT, max_lines=99, gap=2):
        font = font or self.f_s
        for ln in wrap(s, font, width)[:max_lines]:
            self.txt(ln, (x, y), font, color)
            y += font.get_height() + gap
        return y

    def button(self, rect, label, cb, big=False, active=False, dim=False):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        fill = (74, 58, 18) if active else ((46, 84, 128) if hover else (30, 56, 90))
        pygame.draw.rect(self.screen, fill, rect, border_radius=8)
        pygame.draw.rect(self.screen, ACCENT if active else GRID, rect, 1, border_radius=8)
        color = ACCENT if active else (DIM if dim else TEXT)
        lab = (self.f_m if big else self.f_s).render(label, True, color)
        self.screen.blit(lab, lab.get_rect(center=rect.center))
        self.hot.append((rect, cb))

    def chips(self, x, y, w, options, active, click, dim=False):
        cx = x
        for opt in options:
            tw = self.f_s.size(str(opt))[0] + 20
            if cx + tw > x + w:
                cx, y = x, y + 32
            self.button(pygame.Rect(cx, y, tw, 26), str(opt), (lambda o=opt: click(o)),
                        active=active(opt), dim=dim)
            cx += tw + 8
        return y + 34

    def card(self, rect, title):
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=12)
        pygame.draw.rect(self.screen, GRID, rect, 1, border_radius=12)
        self.txt(title, (rect.x + 20, rect.y + 14), self.f_b, ACCENT)

    @staticmethod
    def toggle(s, o):
        s.symmetric_difference_update({o})

    # ================================================================== menu
    def cfg_block(self, cfg, x, y, w, compact=False):
        self.txt("Shooting brain", (x, y), self.f_s, DIM)
        y = self.chips(x, y + 20, w, BRAINS, lambda o: cfg["brain"] == o,
                       lambda o: cfg.__setitem__("brain", o))
        self.txt("Fleet hiding", (x, y), self.f_s, DIM)
        y = self.chips(x, y + 20, w, HIDERS, lambda o: cfg["hider"] == o,
                       lambda o: cfg.__setitem__("hider", o))
        y = self.para(BRAIN_TEXT[cfg["brain"]], x, y + 2, w, self.f_s, DIM)
        if not compact:
            y = self.para(HIDE_TEXT[cfg["hider"]], x, y + 4, w, self.f_s, DIM)
        return y

    def draw_menu(self):
        s = self.screen
        s.fill(BG)
        self.txt("BATTLESHIP AI", (60, 22), self.f_t)
        self.txt("Choose how the AI thinks and how it hides its fleet.", (62, 66), self.f_s, DIM)

        self.card(pygame.Rect(60, 100, 540, 645), "PLAY VS AI")
        self.cfg_block(self.cfg["play"], 90, 150, 480)
        yy = 470
        self.txt("In a game, press:", (90, yy), self.f_m, TEXT)
        yy += 30
        for key, what in [("W", "why the AI fired there"), ("/", "ask the logic engine"),
                          ("H", "heatmap on/off"), ("Esc", "back to menu")]:
            self.txt(key, (90, yy), self.f_s, ACCENT)
            self.txt(what, (140, yy), self.f_s, DIM)
            yy += 22
        self.button(pygame.Rect(90, 690, 480, 40), "Start game", self.start_play, big=True)

        self.card(pygame.Rect(620, 100, 540, 645), "AI VS AI")
        self.txt("AI A", (650, 148), self.f_b, BLUE_A)
        y = self.cfg_block(self.cfg["A"], 650, 176, 480, compact=True)
        self.txt("AI B", (650, y + 14), self.f_b, ORANGE_B)
        self.cfg_block(self.cfg["B"], 650, y + 42, 480, compact=True)
        self.button(pygame.Rect(650, 690, 480, 40), "Watch AI vs AI", self.start_arena, big=True)

    # ================================================================== play
    def start_play(self):
        self.game = Game(self.profile, random.Random(), cfg=copy.deepcopy(self.cfg["play"]))
        self.scene, self.overlay, self.ai_stage = "play", None, 0

    def begin_if_ready(self):
        if not self.game.pending and self.game.phase == "place":
            self.game.start_battle()

    def update_play(self):
        g = self.game
        if g.phase == "battle" and g.turn == "ai":
            now = time.time()
            if self.ai_stage == 0:
                self.ai_due, self.ai_stage = now + 0.45, 1
            elif self.ai_stage == 1 and now >= self.ai_due:
                self.ai_stage = 2                    # draw a "thinking" frame first
            elif self.ai_stage == 2:
                g.ai_move()
                self.ai_stage = 0

    # ================================================================= arena
    def start_arena(self):
        self.sa, self.sb = Side("A", copy.deepcopy(self.cfg["A"])), Side("B", copy.deepcopy(self.cfg["B"]))
        self.arena_rng = random.Random()
        self.records, self.series_done, self.arena_paused = [], False, False
        self.view, self.view_lock, self.last_ev = "A", False, None
        self.scene, self.overlay = "arena", None
        self.new_arena_game()

    def new_arena_game(self):
        first = "A" if len(self.records) % 2 == 0 else "B"
        self.match = Match(self.sa, self.sb, self.arena_rng, samples=100, first=first)
        self.stage, self.next_t, self.last_ev = 0, time.time() + 0.6, None

    def arena_step(self):
        m = self.match
        ev = m.step()
        if ev is None:
            return
        self.last_ev = ev
        if not self.view_lock:
            self.view = ev["side"]
        if m.winner:
            m.finish()
            self.records.append({"winner": m.winner, "shots_a": m.shots_by("A"), "shots_b": m.shots_by("B")})
            self.next_t = time.time() + 1.6

    def update_arena(self):
        if self.series_done or self.arena_paused or self.match is None:
            return
        now, m = time.time(), self.match
        if m.winner:
            if now >= self.next_t:
                if len(self.records) >= self.games_target:
                    self.series_done = True
                else:
                    self.new_arena_game()
            return
        if self.stage == 0:
            self.next_t, self.stage = now + self.delay, 1
        elif self.stage == 1 and now >= self.next_t:
            self.stage = 2
        elif self.stage == 2:
            self.arena_step()
            self.stage = 0

    # ============================================================ inspectors
    def inspector_source(self):
        """(ai, profile, last decision, label) for the overlays, or None."""
        if self.scene == "play" and self.game and self.game.ai:
            return self.game.ai, self.game.profile, self.game.last_ai, "the AI"
        if self.scene == "arena" and self.sa:
            side = self.sa if self.view == "A" else self.sb
            return side.ai, side.profile, side.ai.last, f"AI {self.view}"
        return None

    def open_overlay(self, kind):
        src = self.inspector_source()
        if src is None:
            return
        ai, profile, d, label = src
        self.overlay = kind
        L = []
        if kind == "why":
            L.append((f"WHY DID {label.upper()} FIRE THERE?", "title"))
            if d is None:
                L.append(("It has not moved yet.", "text"))
            else:
                for ln in d["lines"][:7]:
                    L.append((ln, "text"))
                L.append((f"Brain: {d['brain']}   mode: {d['mode']}   thinking time: {d['ms']:.0f} ms", "dim"))
                if d.get("steps"):
                    L.append(("How it decided:", "head"))
                    L += [("  " + x, "text") for x in d["steps"]]
                if d.get("table"):
                    L.append(("Its top candidates (the fired cell is highlighted):", "head"))
                    first = "Layouts say" if d["brain"] == "csp" else "Score (100 = best)"
                    L.append((["Cell", first, "", "Final score", ""], "rowhead"))
                    for cell, p, odds, score in d["table"][:6]:
                        L.append(([cell_name(*cell), f"{p:.0%}", "", f"{score:.0%}",
                                   "<- fired" if cell == d["cell"] else ""],
                                  "rowhot" if cell == d["cell"] else "row"))
                if d["detail"]:
                    L.append(("Rules that fired in its knowledge base (propositional logic):", "head"))
                    L += [("  " + x, "dim") for x in d["detail"][:3]]
            eng = fol.engine_for(ai.kb)
            derived = eng.forward()
            by = Counter(f[0] for f in derived)
            L.append(("First-order logic view: forward chaining over Horn rules with variables:", "head"))
            L.append((f"  {len(derived)} facts derived: " +
                      ", ".join(f"{n} {p}" for p, n in sorted(by.items())), "dim"))
            for pred in ("ForcedShip", "LikelyShip"):
                cells = sorted(f[1] for f in derived if f[0] == pred)
                if cells:
                    L.append((f"  {pred}: {', '.join(cells)}", "text"))
            L.append(("  Press / to ask a question and see the proof tree.", "dim"))
        else:                                                     # ask
            self.ask_text, self.ask_out = "", [("Type a question and press Enter.", "dim")]
            L = []
        self.ovr_lines = L

    def run_ask(self):
        src = self.inspector_source()
        if src is None:
            return
        ai = src[0]
        out = []
        try:
            res = fol.engine_for(ai.kb).ask(self.ask_text)
            goal = fol.fmt(res["goal"])
            if not res["answers"]:
                out.append((f"{goal}  ->  cannot be proved from what the AI knows ({res['steps']} inference steps).", "bad"))
            for a in res["answers"][:4]:
                b = ", ".join(f"{k} = {v}" for k, v in a["bindings"].items()) or "TRUE"
                out.append((f"{goal}  ->  {b}   ({res['steps']} inference steps)", "good"))
                out += [("   " + ln, "dim") for ln in a["proof"][:10]]
        except ValueError as ex:
            out.append((str(ex), "bad"))
        self.ask_out = out

    # =================================================================== input
    def on_key(self, e):
        key = e.key
        if self.overlay == "ask":
            if key == pygame.K_ESCAPE:
                self.overlay = None
            elif key == pygame.K_RETURN:
                self.run_ask()
            elif key == pygame.K_BACKSPACE:
                self.ask_text = self.ask_text[:-1]
            elif e.unicode and e.unicode.isprintable() and e.unicode != "/" and len(self.ask_text) < 40:
                self.ask_text += e.unicode
            return
        if key == pygame.K_ESCAPE:
            if self.overlay:
                self.overlay = None
            elif self.scene != "menu":
                self.scene = "menu"
            else:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            return
        if self.scene in ("play", "arena"):
            if key == pygame.K_w:
                self.open_overlay("why")
            elif key == pygame.K_SLASH:
                self.open_overlay("ask")
            elif key == pygame.K_h:
                self.show_heat = not self.show_heat
        if self.scene == "play":
            g = self.game
            if key == pygame.K_e:
                self.show_enemy = not self.show_enemy
            elif key == pygame.K_n:
                self.start_play()
            elif g.phase == "place":
                if key == pygame.K_r:
                    g.horizontal = not g.horizontal
                elif key == pygame.K_u:
                    g.undo()
                elif key == pygame.K_a:
                    g.auto_place()
                    self.begin_if_ready()
        elif self.scene == "arena":
            if key == pygame.K_SPACE:
                self.arena_paused = not self.arena_paused
            elif key == pygame.K_s and not self.match.winner:
                self.arena_paused = True
                self.arena_step()
            elif key == pygame.K_UP:
                self.delay = max(0.0, self.delay - 0.1) if self.delay <= 0.5 else self.delay - 0.25
            elif key == pygame.K_DOWN:
                self.delay = min(2.0, self.delay + (0.1 if self.delay < 0.5 else 0.25))
            elif key == pygame.K_TAB:
                self.view = "B" if self.view == "A" else "A"
                self.view_lock = True
            elif key == pygame.K_n:
                if self.series_done:
                    self.start_arena()
                else:
                    while not self.match.winner:
                        self.arena_step()
            elif key == pygame.K_g:
                self.games_target += 10
                self.series_done = False

    def on_click(self, e):
        if e.button == 1:
            for rect, cb in reversed(self.hot):
                if rect.collidepoint(e.pos):
                    cb()
                    return
        if self.scene == "play" and self.overlay is None:
            g = self.game
            if g.phase == "place":
                if e.button == 3:
                    g.horizontal = not g.horizontal
                elif e.button == 1:
                    cell = self.cell_at(LEFT, e.pos)
                    if cell and g.place(*cell):
                        self.begin_if_ready()
            elif g.phase == "battle" and g.turn == "player" and e.button == 1:
                cell = self.cell_at(RIGHT, e.pos)
                if cell and g.player_fire(*cell):
                    self.ai_stage = 0

    @staticmethod
    def cell_at(origin, pos):
        x, y = pos[0] - origin[0], pos[1] - origin[1]
        if 0 <= x < BOARD_PX and 0 <= y < BOARD_PX:
            return y // CELL, x // CELL
        return None

    # ============================================================ board drawing
    def cell_rect(self, origin, r, c):
        return pygame.Rect(origin[0] + c * CELL, origin[1] + r * CELL, CELL, CELL)

    def draw_board(self, origin, board, reveal_all, heat=None, hover=None, ghost=None, last=None):
        s = self.screen
        pygame.draw.rect(s, WATER, (*origin, BOARD_PX, BOARD_PX))
        if board:
            for ship in board.ships:
                if reveal_all or ship.sunk:
                    for (r, c) in ship.cells:
                        col = SHIP_SUNK if ship.sunk else SHIP
                        pygame.draw.rect(s, col, self.cell_rect(origin, r, c).inflate(-4, -4), border_radius=7)
        if ghost:
            cells, ok = ghost
            for (r, c) in cells:
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                    surf.fill((96, 214, 130, 150) if ok else (240, 96, 90, 150))
                    s.blit(surf, self.cell_rect(origin, r, c))
        if heat and board:
            pool = [heat[idx(r, c)] for r in range(SIZE) for c in range(SIZE) if (r, c) not in board.shots]
            top = max(pool) if pool else 0
            if top > 0:
                ranked = []
                for r in range(SIZE):
                    for c in range(SIZE):
                        if (r, c) in board.shots:
                            continue
                        p = heat[idx(r, c)]
                        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                        surf.fill((255, 130, 0, int(18 + 150 * p / top)))
                        s.blit(surf, self.cell_rect(origin, r, c))
                        ranked.append((p, r, c))
                ranked.sort(reverse=True)
                for p, r, c in ranked[:6]:
                    lab = self.f_s.render(f"{p * 100:.0f}", True, (255, 255, 255))
                    s.blit(lab, lab.get_rect(center=self.cell_rect(origin, r, c).center))
        for i in range(SIZE + 1):
            pygame.draw.line(s, GRID, (origin[0] + i * CELL, origin[1]), (origin[0] + i * CELL, origin[1] + BOARD_PX))
            pygame.draw.line(s, GRID, (origin[0], origin[1] + i * CELL), (origin[0] + BOARD_PX, origin[1] + i * CELL))
        for i in range(SIZE):
            a = self.f_s.render(COLS[i], True, DIM)
            s.blit(a, a.get_rect(center=(origin[0] + i * CELL + CELL // 2, origin[1] - 12)))
            b = self.f_s.render(str(i + 1), True, DIM)
            s.blit(b, b.get_rect(center=(origin[0] - 14, origin[1] + i * CELL + CELL // 2)))
        if board:
            for (r, c), res in board.shots.items():
                rect = self.cell_rect(origin, r, c)
                if res == "miss":
                    pygame.draw.circle(s, MISS, rect.center, 5)
                else:
                    pygame.draw.circle(s, HIT, rect.center, 12)
                    if board.occupied[(r, c)].sunk:
                        pygame.draw.line(s, (255, 255, 255), (rect.left + 10, rect.top + 10),
                                         (rect.right - 10, rect.bottom - 10), 2)
                        pygame.draw.line(s, (255, 255, 255), (rect.left + 10, rect.bottom - 10),
                                         (rect.right - 10, rect.top + 10), 2)
        if last:
            pygame.draw.rect(s, ACCENT, self.cell_rect(origin, *last), 3)
        if hover:
            pygame.draw.rect(s, ACCENT, self.cell_rect(origin, *hover), 2)

    def draw_fleet_status(self, x, y, title, board, show_hits):
        self.txt(title, (x, y), self.f_m, DIM)
        for i, (name, length) in enumerate(SHIPS):
            ship = next((sh for sh in board.ships if sh.name == name), None) if board else None
            yy = y + 24 + i * 22
            self.txt(name, (x, yy), self.f_s, DIM if (ship and ship.sunk) else TEXT)
            for k in range(length):
                col = (70, 84, 100)
                if ship is not None and ship.sunk:
                    col = HIT
                elif ship is not None and show_hits and k < len(ship.hits):
                    col = HIT
                pygame.draw.rect(self.screen, col, (x + 100 + k * 17, yy + 2, 14, 12), border_radius=3)

    # ================================================================== panel
    def draw_panel(self, title, d, profile, hide_info, tag, model_title, brain=None):
        pygame.draw.rect(self.screen, PANEL_BG, PANEL, border_radius=10)
        pygame.draw.rect(self.screen, GRID, PANEL, 1, border_radius=10)
        x, w, y = PANEL.x + 14, PANEL.w - 28, PANEL.y + 8
        self.txt(title, (x, y), self.f_b, ACCENT)
        y += 28
        self.txt(tag, (x, y), self.f_s, DIM)
        y += 22
        limit = PANEL.y + 430
        if d is None:
            y = self.para("Waiting for the first shot. Each turn the AI updates its knowledge, "
                          "solves for the possible fleet layouts and picks a cell. Press W for the full reasoning.",
                          x, y, w, self.f_s, DIM)
        else:
            col = HIT if d["mode"] == "TARGET" else GOOD
            self.txt(f"Mode: {d['mode']}", (x, y), self.f_m, col)
            y += 24
            if d["brain"] == "csp":
                kind = "exact count" if d["exact"] else "sampled"
                self.txt(f"Layouts: {d['n']} ({kind})", (x, y), self.f_s)
                y += 18
                self.txt(f"CSP nodes: {d['nodes']}   time: {d['ms']:.0f} ms", (x, y), self.f_s)
                y += 18
            elif d["brain"] in ("logic", "density"):
                self.txt(f"time: {d['ms']:.1f} ms", (x, y), self.f_s)
                y += 18
            if d["brain"] in ("csp", "logic"):
                self.txt(f"Dead cells: {d['dead']}   Forced cells: {d['forced']}", (x, y), self.f_s)
                y += 18
            hk = {"prob": "Heatmap = probability of a ship (%)",
                  "relative": "Heatmap = relative score (100 = best cell)"}.get(d["heat_kind"],
                                                                                "No heatmap for this brain")
            self.txt(hk, (x, y), self.f_s, DIM)
            y += 24
            for ln in d["lines"]:
                y = self.para(ln, x, y, w, self.f_s, TEXT, max_lines=5, gap=1) + 5
                if y > limit - 60:
                    break
            if d["top"] and d["brain"] in ("csp", "density"):
                y += 2
                for cell, p in d["top"][:3]:
                    if y > limit + 12:
                        break
                    self.txt(cell_name(*cell), (x, y), self.f_s, TEXT)
                    pygame.draw.rect(self.screen, (40, 66, 100), (x + 34, y + 3, 150, 10), border_radius=4)
                    pygame.draw.rect(self.screen, ACCENT, (x + 34, y + 3, max(2, int(150 * min(1, p))), 10), border_radius=4)
                    self.txt(f"{p:.0%}", (x + 192, y), self.f_s, DIM)
                    y += 18
        y = max(y + 6, limit)
        pygame.draw.line(self.screen, GRID, (x, y), (x + w, y))
        y += 6
        self.txt("FLEET HIDING", (x, y), self.f_b, ACCENT)
        y += 28
        if hide_info:
            y = self.para(hide_info["text"], x, y + 2, w, self.f_s, DIM, max_lines=6)
        self.txt("W why   / ask", (x, PANEL.bottom - 24), self.f_s, DIM)

    # ================================================================ overlays
    def draw_overlay(self):
        if not self.overlay:
            return
        r = OVR
        pygame.draw.rect(self.screen, (9, 18, 31), r, border_radius=12)
        pygame.draw.rect(self.screen, ACCENT, r, 2, border_radius=12)
        x, y, w = r.x + 20, r.y + 14, r.w - 40
        if self.overlay == "ask":
            self.txt("ASK THE AI'S LOGIC ENGINE (first-order logic, backward chaining)", (x, y), self.f_b, ACCENT)
            y += 34
            box = pygame.Rect(x, y, w, 34)
            pygame.draw.rect(self.screen, (22, 42, 68), box, border_radius=6)
            pygame.draw.rect(self.screen, GRID, box, 1, border_radius=6)
            cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
            self.txt("> " + self.ask_text + cursor, (x + 10, y + 6), self.f_m, TEXT)
            y += 46
            self.txt("Try:  " + "    ".join(fol.EXAMPLES), (x, y), self.f_s, DIM)
            y += 24
            y = self.para("Variables start with ?  (e.g. ForcedShip(?c) lists every cell the AI can PROVE holds a ship). "
                          "Cells are written like E5. The AI proves it by chaining Horn rules backwards from your question.",
                          x, y, w, self.f_s, DIM)
            lines = self.ask_out
        else:
            lines = self.ovr_lines
            y += 0
        colors = {"title": ACCENT, "head": GOOD, "text": TEXT, "dim": DIM, "good": GOOD, "bad": BAD}
        y += 8
        for text, kind in lines:
            if kind in ("row", "rowhead", "rowhot"):                 # a table row: list of columns
                if y <= r.bottom - 36:
                    col = {"rowhead": DIM, "row": TEXT, "rowhot": ACCENT}[kind]
                    for cx, cell_text in zip((0, 110, 290, 440, 580), text):
                        self.txt(cell_text, (x + 14 + cx, y), self.f_s, col)
                    y += self.f_s.get_height() + 3
                continue
            font = self.f_b if kind == "title" else (self.f_m if kind == "head" else self.f_s)
            indent = (len(text) - len(text.lstrip(" "))) * 6          # keep leading spaces (tree, lists)
            for ln in wrap(text.lstrip(" "), font, w - indent):
                if y > r.bottom - 52:
                    break
                self.txt(ln, (x + indent, y), font, colors.get(kind, TEXT))
                y += font.get_height() + 3
            y += 4 if kind in ("title", "head") else 1
        self.txt("Esc close     W why     / ask", (x, r.bottom - 26), self.f_s, DIM)

    # ================================================================== scenes
    def menu_button(self):
        self.button(pygame.Rect(W - 120, 22, 90, 30), "Menu", lambda: setattr(self, "scene", "menu"))

    def draw_play(self):
        g, s = self.game, self.screen
        s.fill(BG)
        self.txt("BATTLESHIP AI", (60, 22), self.f_t)
        self.txt("You vs " + cfg_label(g.cfg).replace("@", "  -  hides by "), (62, 66), self.f_s, DIM)
        self.menu_button()
        mouse = pygame.mouse.get_pos()
        ghost = hover = None
        if g.phase == "place":
            cell = self.cell_at(LEFT, mouse)
            if cell and g.pending:
                cells = [(cell[0], cell[1] + i) if g.horizontal else (cell[0] + i, cell[1])
                         for i in range(g.pending[0][1])]
                ghost = (cells, g.can_place(*cell))
        elif g.phase == "battle" and g.turn == "player" and self.overlay is None:
            hover = self.cell_at(RIGHT, mouse)
        heat = g.ai.heat if (g.ai and self.show_heat) else None
        self.txt("YOUR FLEET", (LEFT[0], LEFT[1] - 40), self.f_b)
        self.txt("ENEMY WATERS", (RIGHT[0], RIGHT[1] - 40), self.f_b)
        last = g.last_ai["cell"] if g.last_ai else None
        self.draw_board(LEFT, g.player, True, heat=heat, ghost=ghost, last=last)
        self.draw_board(RIGHT, g.enemy, self.show_enemy, hover=hover)
        if g.phase == "place":
            veil = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
            veil.fill((11, 22, 38, 150))
            s.blit(veil, RIGHT)
            lab = self.f_m.render("Place your fleet first", True, TEXT)
            s.blit(lab, lab.get_rect(center=(RIGHT[0] + BOARD_PX // 2, RIGHT[1] + BOARD_PX // 2)))
        if g.phase == "place":
            if g.pending:
                name, length = g.pending[0]
                msg = (f"Place your {name} ({length} cells), {'horizontal' if g.horizontal else 'vertical'}. "
                       "Click to place, R or right-click to rotate, U undo, A auto-place all.")
                self.para(msg, 60, 88, 800, self.f_m, ACCENT)
        else:
            self.txt(g.player_msg, (60, 88), self.f_m, TEXT)
            self.txt(g.ai_msg, (60, 108), self.f_m, DIM)
            if g.phase == "battle" and g.turn == "ai" and self.ai_stage == 2:
                self.txt("AI is thinking...", (480, 108), self.f_m, ACCENT)
        self.draw_fleet_status(LEFT[0], 595, "Your ships", g.player, True)
        self.draw_fleet_status(RIGHT[0], 595, "Enemy ships (red = sunk)", g.enemy, False)
        self.draw_panel("AI BRAIN", g.last_ai, g.profile, g.placement_info, cfg_label(g.cfg),
                        "")
        self.txt("H heatmap   E reveal enemy (debug)   N new game   W why   / ask   Esc menu",
                 (60, 760), self.f_s, DIM)
        if g.phase == "over":
            box = pygame.Rect(0, 0, 520, 150)
            box.center = (480, 400)
            pygame.draw.rect(s, (8, 16, 28), box, border_radius=14)
            pygame.draw.rect(s, ACCENT, box, 2, border_radius=14)
            win = g.winner == "player"
            t = self.f_t.render("YOU WIN!" if win else "THE AI WINS", True, GOOD if win else BAD)
            s.blit(t, t.get_rect(center=(box.centerx, box.y + 48)))
            u = self.f_m.render(f"You fired {len(g.player_shots)} shots.", True, TEXT)
            s.blit(u, u.get_rect(center=(box.centerx, box.y + 92)))
            v = self.f_m.render("Press N for a new game", True, ACCENT)
            s.blit(v, v.get_rect(center=(box.centerx, box.y + 120)))

    def draw_arena(self):
        s, m, a, b = self.screen, self.match, self.sa, self.sb
        s.fill(BG)
        self.txt("AI VS AI", (60, 22), self.f_t)
        self.menu_button()
        wins_a = sum(r["winner"] == "A" for r in self.records)
        wins_b = len(self.records) - wins_a
        self.txt("A " + cfg_label(a.cfg.__class__(a.cfg)), (62, 66), self.f_s, BLUE_A)
        self.txt(f"{wins_a}  :  {wins_b}", (440, 62), self.f_m, TEXT)
        self.txt("B " + cfg_label(b.cfg), (500, 66), self.f_s, ORANGE_B)
        game_no = min(len(self.records) + (0 if m.winner else 1), self.games_target)
        spd = "max" if self.delay <= 0.01 else f"{self.delay:.2f}s/shot"
        state = "PAUSED" if self.arena_paused else ("series complete" if self.series_done else "running")
        self.txt(f"Game {game_no} of {self.games_target}   speed {spd}   {state}", (62, 88), self.f_s, DIM)
        if not m.winner and not self.arena_paused and self.stage == 2:
            self.txt(f"AI {m.turn} is thinking...", (62, 108), self.f_m, ACCENT)
        elif self.last_ev:
            e = self.last_ev
            res = "sank a ship!" if e["sunk"] else ("hit" if e["result"] == "hit" else "miss")
            self.txt(f"AI {e['side']} fired at {cell_name(*e['cell'])}: {res}", (62, 108), self.f_m, TEXT)
        la = self.last_ev["cell"] if (self.last_ev and self.last_ev["side"] == "B") else None
        lb = self.last_ev["cell"] if (self.last_ev and self.last_ev["side"] == "A") else None
        self.txt("AI A's FLEET", (LEFT[0], LEFT[1] - 40), self.f_b, BLUE_A)
        self.txt("AI B's FLEET", (RIGHT[0], RIGHT[1] - 40), self.f_b, ORANGE_B)
        self.draw_board(LEFT, a.board, True, heat=(b.ai.heat if self.show_heat else None), last=la)
        self.draw_board(RIGHT, b.board, True, heat=(a.ai.heat if self.show_heat else None), last=lb)
        self.txt("heat = what B believes", (LEFT[0], LEFT[1] + BOARD_PX + 4), self.f_s, DIM)
        self.txt("heat = what A believes", (RIGHT[0], RIGHT[1] + BOARD_PX + 4), self.f_s, DIM)
        self.draw_shots_chart(60, 600)
        side = a if self.view == "A" else b
        self.draw_panel(f"AI {self.view} BRAIN", side.ai.last, side.profile, side.hide_info,
                        cfg_label(side.cfg), "")
        self.txt("Space pause  S step  Up/Down speed  Tab switch brain  H heat  N next  G +10 games  W / inspect",
                 (60, 765), self.f_s, DIM)
        if self.series_done:
            self.draw_series_summary()

    def draw_shots_chart(self, x, y):
        self.txt("Shots fired by the winner, per game", (x, y), self.f_s, DIM)
        h, w = 118, 790
        base = y + 22 + h
        n = max(self.games_target, len(self.records), 1)
        bw = max(6, min(40, (w - 8) // n - 4))
        pygame.draw.line(self.screen, GRID, (x, base), (x + w, base))
        for i, r in enumerate(self.records[-(w // (bw + 4)):]):
            shots = r["shots_a"] if r["winner"] == "A" else r["shots_b"]
            bh = int(h * min(shots, 100) / 100)
            col = BLUE_A if r["winner"] == "A" else ORANGE_B
            rect = pygame.Rect(x + 4 + i * (bw + 4), base - bh, bw, bh)
            pygame.draw.rect(self.screen, col, rect, border_radius=3)
            if bw >= 20:
                lab = self.f_s.render(str(shots), True, TEXT)
                self.screen.blit(lab, lab.get_rect(midbottom=(rect.centerx, rect.top - 1)))

    def draw_series_summary(self):
        sm = summarise(self.records, self.sa.cfg, self.sb.cfg)
        box = pygame.Rect(0, 0, 560, 230)
        box.center = (470, 380)
        pygame.draw.rect(self.screen, (8, 16, 28), box, border_radius=14)
        pygame.draw.rect(self.screen, ACCENT, box, 2, border_radius=14)
        self.txt("SERIES COMPLETE", (box.x + 24, box.y + 16), self.f_b, ACCENT)
        fa = "-" if sm["mean_win_shots_a"] is None else f"{sm['mean_win_shots_a']:.1f}"
        fb = "-" if sm["mean_win_shots_b"] is None else f"{sm['mean_win_shots_b']:.1f}"
        rows = [(f"AI A   {sm['a']}", BLUE_A), (f"   wins {sm['wins_a']}     avg shots when winning {fa}", TEXT),
                (f"AI B   {sm['b']}", ORANGE_B), (f"   wins {sm['wins_b']}     avg shots when winning {fb}", TEXT)]
        y = box.y + 54
        for text, col in rows:
            self.txt(text, (box.x + 24, y), self.f_m, col)
            y += 28
        self.txt("N new series     G play 10 more games     Esc menu", (box.x + 24, box.bottom - 34), self.f_s, DIM)

    # ================================================================== loop
    def draw(self):
        self.hot = []
        {"menu": self.draw_menu, "play": self.draw_play, "arena": self.draw_arena,
         }[self.scene]()
        self.draw_overlay()

    def update(self):
        if self.scene == "play":
            self.update_play()
        elif self.scene == "arena":
            self.update_arena()

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    self.on_key(e)
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    self.on_click(e)
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()


# ======================================================================
# headless helpers for testing / screenshots
# ======================================================================
def demo(args):
    app = App(history_path=None)
    rng = random.Random(5)
    sc = args.scene
    if sc == "menu":
        pass
    elif sc in ("play", "why", "ask"):
        app.start_play()
        g = app.game
        g.auto_place()
        g.start_battle()
        for _ in range(args.rounds):
            free = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in g.enemy.shots]
            g.player_fire(*rng.choice(free))
            g.ai_move()
            if g.phase != "battle":
                break
        if sc == "why":
            app.open_overlay("why")
        elif sc == "ask":
            app.open_overlay("ask")
            app.ask_text = args.query
            app.run_ask()
    elif sc == "arena":
        app.start_arena()
        steps = 0
        while len(app.records) < args.games and steps < 4000:
            app.arena_step()
            steps += 1
            if app.match.winner:
                if len(app.records) < args.games:
                    app.new_arena_game()
        if len(app.records) < args.games:
            pass
        else:
            app.games_target = args.games
            app.series_done = False
        # leave the next game part-way through for the picture
        if len(app.records) < args.games + 1:
            app.games_target = args.games + 1
            app.new_arena_game()
            for _ in range(args.rounds):
                app.arena_step()
    if sc in ("play", "why", "ask"):
        app.scene = "play"
    elif sc == "arena":
        app.scene = "arena"
    app.draw()
    pygame.image.save(app.screen, args.shot)
    print("saved", args.shot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-shot", metavar="FILE", dest="shot", help="render a scene headlessly (testing)")
    ap.add_argument("--scene", default="play", choices=["menu", "play", "why", "ask", "arena"])
    ap.add_argument("--rounds", type=int, default=14)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--query", default="ForcedShip(?c)")
    args = ap.parse_args()
    if args.shot:
        demo(args)
    else:
        App().run()
