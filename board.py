"""
board.py - Battleship rules: grid, ships and shots.

Cells are (row, col) pairs. For bitmask work (used by the CSP solver) a cell maps
to bit number  row * SIZE + col.
"""
import random
from functools import lru_cache

SIZE = 10
SHIPS = [("Carrier", 5), ("Battleship", 4), ("Cruiser", 3), ("Submarine", 3), ("Destroyer", 2)]
TOTAL_SHIP_CELLS = sum(length for _, length in SHIPS)
COLS = "ABCDEFGHIJ"


def idx(r, c):
    return r * SIZE + c


def cell_name(r, c):
    return f"{COLS[c]}{r + 1}"


def in_bounds(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def neighbors(r, c):
    """Orthogonal neighbours that are inside the board."""
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc):
            yield nr, nc


def ship_cells(length, r, c, horizontal):
    return [(r, c + i) if horizontal else (r + i, c) for i in range(length)]


@lru_cache(maxsize=None)
def placements(length):
    """Every legal placement of a ship of this length on an empty board.

    Returns a tuple of (bitmask, cells, row, col, horizontal).
    These are the *domain values* of the CSP.
    """
    out = []
    for r in range(SIZE):
        for c in range(SIZE):
            for horizontal in (True, False):
                cells = ship_cells(length, r, c, horizontal)
                if all(in_bounds(a, b) for a, b in cells):
                    mask = 0
                    for a, b in cells:
                        mask |= 1 << idx(a, b)
                    out.append((mask, tuple(cells), r, c, horizontal))
    return tuple(out)


class Ship:
    def __init__(self, name, length, r, c, horizontal):
        self.name = name
        self.length = length
        self.r, self.c, self.horizontal = r, c, horizontal
        self.cells = ship_cells(length, r, c, horizontal)
        self.hits = set()

    @property
    def sunk(self):
        return len(self.hits) == self.length


class Board:
    def __init__(self):
        self.ships = []
        self.occupied = {}   # (r, c) -> Ship
        self.shots = {}      # (r, c) -> "hit" | "miss"

    # ---- placement -------------------------------------------------------
    def can_place(self, length, r, c, horizontal):
        cells = ship_cells(length, r, c, horizontal)
        return all(in_bounds(a, b) and (a, b) not in self.occupied for a, b in cells)

    def place(self, name, length, r, c, horizontal):
        if not self.can_place(length, r, c, horizontal):
            return None
        ship = Ship(name, length, r, c, horizontal)
        self.ships.append(ship)
        for cell in ship.cells:
            self.occupied[cell] = ship
        return ship

    def remove_last(self):
        if not self.ships:
            return None
        ship = self.ships.pop()
        for cell in ship.cells:
            del self.occupied[cell]
        return ship

    def place_fleet(self, fleet):
        """fleet: list of (name, length, r, c, horizontal)."""
        for name, length, r, c, h in fleet:
            if self.place(name, length, r, c, h) is None:
                raise ValueError("illegal fleet")

    def fleet_description(self):
        return [(s.name, s.length, s.r, s.c, s.horizontal) for s in self.ships]

    @classmethod
    def random(cls, rng=random):
        board = cls()
        for name, length in SHIPS:
            while True:
                _, _, r, c, h = rng.choice(placements(length))
                if board.can_place(length, r, c, h):
                    board.place(name, length, r, c, h)
                    break
        return board

    # ---- shooting --------------------------------------------------------
    def fire(self, r, c):
        """Returns (result, ship). result is 'hit', 'miss' or 'repeat'.
        When the shot sinks a ship, ship.sunk is True."""
        if (r, c) in self.shots:
            return "repeat", None
        ship = self.occupied.get((r, c))
        if ship is None:
            self.shots[(r, c)] = "miss"
            return "miss", None
        ship.hits.add((r, c))
        self.shots[(r, c)] = "hit"
        return "hit", ship

    def all_sunk(self):
        return bool(self.ships) and all(s.sunk for s in self.ships)
