# ⚓ BATTLESHIP AI
**Python + Pygame — Logic, Constraint Satisfaction and Probability — Dark Naval Theme**

Play Battleship against an AI that reasons like a detective, or watch two AIs play each other.
Every shot the AI fires can be explained: the rules it used, the layouts it considered and the cells it compared.

---

## ✅ STEP-BY-STEP: Running in VS Code

### STEP 1 — Install Prerequisites
- **Python 3.10 or higher:** https://www.python.org/downloads/
- After installing, verify with `py --version` (Windows) or `python3 --version` (macOS/Linux) in the terminal.
- **VS Code:** https://code.visualstudio.com/

### STEP 2 — Install the VS Code Extension
1. Open VS Code
2. Press `Ctrl+Shift+X` (Extensions panel)
3. Search for **"Python"** by Microsoft
4. Click **Install**

### STEP 3 — Open the Project
1. Open VS Code
2. Click **File → Open Folder**
3. Select the `battleship_ai` folder
4. Open a terminal with `` Ctrl+` ``

### STEP 4 — Install pygame (one time)
```
# Windows (PowerShell)
py -m pip install -r requirements.txt

# macOS / Linux
python3 -m pip install -r requirements.txt
```

### STEP 5 — Run the Game
**Option A — Click Run (easiest):**
- Open `main.py` and click the ▶ **Run Python File** button at the top right.

**Option B — Terminal:**
```
# Windows
py selftest.py      # optional: should end with "All 59 checks passed."
py main.py

# macOS / Linux
python3 selftest.py
python3 main.py
```

**Option C — Double-click:** `run_game.bat` (Windows) or `run_game.sh` (macOS / Linux).

---

## 🗂️ PROJECT STRUCTURE
```
battleship_ai/
├── main.py              ← MAIN GAME (run this): Pygame UI, menu, overlays
├── game.py              ← You-vs-AI game flow
├── arena.py             ← AI-vs-AI matches and series (also a command-line tool)
├── board.py             ← Rules, ships, legal placements (bitmasks)
├── ai_player.py         ← The five shooting brains + explanations
├── knowledge_base.py    ← Facts and forward-chaining rules R1–R6
├── resolution.py        ← Propositional resolution prover
├── fol.py               ← First-order logic engine (backward chaining)
├── csp_solver.py        ← Hidden fleet as a Constraint Satisfaction Problem
├── probability.py       ← Layouts → probability heatmap
├── hiding.py            ← One entry point for fleet-hiding methods
├── placement.py         ← Random / hill climbing / genetic algorithm hiding
├── duel.py              ← Minimax + alpha-beta pruning hiding duel
├── player_model.py, layout.py, habits.py   ← Helpers used by the AI core and tests
├── benchmark.py         ← Shots-per-brain benchmark
├── selftest.py          ← 59 automated correctness checks
├── docs/                ← Screenshots
├── requirements.txt     ← pygame
└── README.md            ← This file
```

---

## 🎮 HOW TO PLAY

| Screen | What to do |
|--------|-----------|
| **MENU** | Pick the AI's shooting brain and fleet-hiding method, then press *Start game* or *Watch AI vs AI* |
| **PLACE FLEET** | Click to place each ship. `R` / right-click rotates, `U` undoes, `A` auto-places |
| **BATTLE** | Click a cell on *Enemy Waters* to fire. The AI answers after each shot |
| **WHY PANEL** | Press `W` to see why the AI fired where it did |
| **ASK THE AI** | Press `/` and type a logic question, e.g. `ForcedShip(?c)` or `Ship(E5)` |
| **HEATMAP** | Press `H` to show or hide the AI's belief about where ships are |
| **AI VS AI** | `Space` pause · `S` step · `↑/↓` speed · `Tab` switch AI · `N` next game · `G` +10 games |

Other keys: `E` reveal enemy fleet (debug) · `N` new game · `Esc` back to menu.

---

## 🧠 SHOOTING BRAINS

| Brain | How it chooses a cell |
|-------|-----------------------|
| `random` | Random unknown cell (baseline) |
| `hunt` | Hunt/target with checkerboard parity |
| `logic` | Forward chaining + resolution only; explains every step |
| `density` | Counts legal ship placements through each cell |
| `csp` | Solves the fleet as a CSP and fires at the most probable cell |

**Fleet hiding:** `random` · `hill` (hill climbing with restarts) · `genetic` (genetic algorithm) · `minimax` (minimax + alpha-beta duel)

### 🔑 THE LOGIC RULES
| Rule | Meaning |
|------|---------|
| R1 | A miss means no ship on that cell |
| R2 | A hit means a ship on that cell |
| R3 | Cells of a sunk ship are used up |
| R4 | A cell in a gap too small for the smallest remaining ship is dead |
| R5 | A hit needs a ship neighbour; resolution proves which one when the rest are empty |
| R6 | Hits in a line suggest the ship continues (heuristic only) |

---

## 🧱 HOW THE AI THINKS (one turn)
```
Shot result → update knowledge base → forward chaining (R1–R6)
            → CSP over all consistent fleet layouts → probability heatmap
            → fire at the highest-score cell
```
- **CSP:** variables = ships still afloat · domains = legal placements · constraints = no overlap, cover every hit, an afloat ship has an unhit cell.
- **Search:** backtracking with MRV, value ordering, forward checking and coverage pruning; exact count when possible, sampling otherwise.

---

## 📊 MEASURED RESULTS
Mean shots to sink the fleet on identical fleets (lower is better, 100 is worst):

| Brain | Mean shots |
|-------|-----------|
| csp | 44.9 (±1.2) |
| density | 45.1 (±0.7) |
| logic | 50.5 |
| hunt | 55.8 |
| random | 94.6 |

`csp` and `density` are a statistical tie; the CSP's advantage is exact reasoning and full explanations.
In 20-game AI-vs-AI series, `csp` beat `hunt` 15–5 and `density` 13–7.

### Command-line tools
```
python benchmark.py                                    # shots per brain (about 1 minute)
python arena.py --a "csp@genetic" --b "density@hill" --games 20
python selftest.py                                     # 59 correctness checks
