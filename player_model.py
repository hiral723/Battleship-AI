"""
player_model.py - the AI learns *you*.

Three learning pieces, all written from scratch (no scikit-learn needed):

1. DecisionTree (ID3, entropy)  - supervised learning.
   Learns "is there a ship on this cell?" from where YOU placed ships in past
   games, using simple cell features (ring distance from the edge, corner,
   parity, board half). The AI uses the prediction to bias its shots.

2. K-means clustering - unsupervised learning.
   Each past game becomes a point (edge_frac, touch_frac, horiz_frac). Centroids
   start at four archetypes and move to fit your games; the nearest centroid
   names your placement style.

3. Shot-habit weights.
   Counts where you tend to fire (early shots count more). The AI places its own
   fleet away from those cells (see placement.py).

History is stored in data/player_history.json.
"""
import json
import math
import os

from board import SIZE, TOTAL_SHIP_CELLS, neighbors, ship_cells
from layout import style_log_weight, mask_of, random_stats

PRIOR_GAMES = 4             # pseudo-games of "random player" mixed into every learned style
STYLE_MIN_EVIDENCE = 1.0    # nats of evidence before the UI names a style (else "No clear habit")
STYLE_FULL_EVIDENCE = 2.0   # nats of evidence at which the style re-weights layouts at full strength


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def ring(r, c):
    return min(r, c, SIZE - 1 - r, SIZE - 1 - c)


def cell_features(r, c):
    corner = ring(r, c) == 0 and r in (0, SIZE - 1) and c in (0, SIZE - 1)
    return {
        "ring": str(min(ring(r, c), 3)),
        "corner": str(corner),
        "parity": str((r + c) % 2),
        "top_half": str(r < SIZE // 2),
        "left_half": str(c < SIZE // 2),
    }


ARCHETYPES = {                      # (edge_frac, touch_frac, horiz_frac)
    "Edge hugger": [0.75, 0.30, 0.50],
    "Clumper": [0.25, 0.85, 0.50],
    "Spreader": [0.20, 0.05, 0.50],
    "Horizontal lover": [0.30, 0.30, 0.95],
}


def game_features(placement):
    """placement: list of (name, length, r, c, horizontal)."""
    owner = {}
    for i, (_, length, r, c, h) in enumerate(placement):
        for cell in ship_cells(length, r, c, h):
            owner[cell] = i
    edge = sum(1 for (r, c) in owner if ring(r, c) == 0) / max(1, len(owner))
    touching = set()
    for (r, c), i in owner.items():
        for n in neighbors(r, c):
            if n in owner and owner[n] != i:
                touching.add(i)
    touch = len(touching) / max(1, len(placement))
    horiz = sum(1 for p in placement if p[4]) / max(1, len(placement))
    return [edge, touch, horiz]


# --------------------------------------------------------------------------
# Decision tree (ID3)
# --------------------------------------------------------------------------
def _entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    p = sum(labels) / n
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


class DecisionTree:
    def __init__(self, max_depth=3, min_samples=8):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def fit(self, rows, labels):
        self.root = self._build(rows, labels, 0)
        return self

    def _leaf(self, labels):
        n, pos = len(labels), sum(labels)
        return {"leaf": (pos + 1) / (n + 2), "n": n}       # Laplace smoothing

    def _build(self, rows, labels, depth):
        n, pos = len(labels), sum(labels)
        if depth >= self.max_depth or n < self.min_samples or pos in (0, n):
            return self._leaf(labels)
        base, best = _entropy(labels), None
        for feat in rows[0]:
            groups = {}
            for row, y in zip(rows, labels):
                groups.setdefault(row[feat], []).append(y)
            if len(groups) < 2:
                continue
            split = sum(len(g) / n * _entropy(g) for g in groups.values())
            gain = base - split                              # information gain
            if best is None or gain > best[0]:
                best = (gain, feat)
        if best is None or best[0] < 1e-4:
            return self._leaf(labels)
        feat = best[1]
        children = {}
        for value in {row[feat] for row in rows}:
            sub = [(r, y) for r, y in zip(rows, labels) if r[feat] == value]
            children[value] = self._build([s[0] for s in sub], [s[1] for s in sub], depth + 1)
        return {"feat": feat, "children": children, "default": (pos + 1) / (n + 2)}

    def predict(self, row):
        node = self.root
        while "leaf" not in node:
            child = node["children"].get(row[node["feat"]])
            if child is None:
                return node["default"]
            node = child
        return node["leaf"]

    def describe(self, node=None, indent=0):
        node = node or self.root
        pad = "  " * indent
        if "leaf" in node:
            return [f"{pad}P(ship)={node['leaf']:.2f} (n={node['n']})"]
        lines = []
        for value, child in sorted(node["children"].items()):
            lines.append(f"{pad}if {node['feat']} = {value}:")
            lines += self.describe(child, indent + 1)
        return lines


# --------------------------------------------------------------------------
# K-means
# --------------------------------------------------------------------------
def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def nearest(point, centroids):
    return min(range(len(centroids)), key=lambda j: _dist(point, centroids[j]))


def kmeans(points, centroids, iters=25):
    centroids = [list(c) for c in centroids]
    for _ in range(iters):
        clusters = [[] for _ in centroids]
        for p in points:
            clusters[nearest(p, centroids)].append(p)
        new = []
        for j, cl in enumerate(clusters):
            if cl:
                new.append([sum(p[k] for p in cl) / len(cl) for k in range(len(cl[0]))])
            else:
                new.append(centroids[j])
        if new == centroids:
            break
        centroids = new
    return centroids


# --------------------------------------------------------------------------
# Player profile: history + trained models
# --------------------------------------------------------------------------
class PlayerProfile:
    def __init__(self, path=None):
        self.path = path
        self.games = []
        self.tree = None
        self.centroids = None
        self.style = None
        self._shot_counts = None
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self.games = json.load(f).get("games", [])
            except (OSError, ValueError):
                self.games = []
        self.retrain()

    # ---- bookkeeping ----
    def add_game(self, placement, shots, winner):
        self.games.append({"placement": [list(p) for p in placement],
                           "shots": [list(s) for s in shots], "winner": winner})
        self.retrain()
        self.save()

    def save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump({"games": self.games}, f)
        except OSError:
            pass

    @property
    def confidence(self):
        n = len(self.games)
        return n / (n + 4)

    # ---- training ----
    def retrain(self):
        n = len(self.games)
        if n == 0:
            self.tree = self.centroids = self.style = None
            self._shot_counts = None
            return
        rows, labels = [], []
        for g in self.games:
            occupied = set()
            for _, length, r, c, h in g["placement"]:
                occupied.update(ship_cells(length, r, c, bool(h)))
            for r in range(SIZE):
                for c in range(SIZE):
                    rows.append(cell_features(r, c))
                    labels.append(1 if (r, c) in occupied else 0)
        self.tree = DecisionTree().fit(rows, labels)

        feats = [game_features(g["placement"]) for g in self.games]
        self.centroids = kmeans(feats, list(ARCHETYPES.values()))
        names = list(ARCHETYPES)
        votes = [nearest(f, self.centroids) for f in feats]
        self.style = names[max(set(votes), key=votes.count)]

        counts = [[0.0] * SIZE for _ in range(SIZE)]
        for g in self.games:
            for i, (r, c) in enumerate(g["shots"]):
                counts[r][c] += 1.0 / (1.0 + i / 8.0)         # early shots matter more
        self._shot_counts = counts

    # ---- what the AI uses ----
    def cell_bias(self, r, c):
        """Multiplier (>1 = you like putting ships here) applied to the AI's
        shot score. 1.0 until the AI has seen some of your games."""
        if self.tree is None:
            return 1.0
        base = TOTAL_SHIP_CELLS / (SIZE * SIZE)
        ratio = self.tree.predict(cell_features(r, c)) / base
        return max(0.4, min(2.5, ratio ** (1.5 * self.confidence)))

    def shot_weights(self):
        """SIZE x SIZE weights (mean ~1): cells you like to shoot at.
        With no history, assume the usual human habit of shooting near the centre."""
        prior = [[1.0 + 0.5 * (ring(r, c) / 4.5) for c in range(SIZE)] for r in range(SIZE)]
        if self._shot_counts is None:
            return prior
        total = sum(sum(row) for row in self._shot_counts) + 0.5 * SIZE * SIZE
        conf = self.confidence
        out = []
        for r in range(SIZE):
            row = []
            for c in range(SIZE):
                emp = (self._shot_counts[r][c] + 0.5) / (total / (SIZE * SIZE))
                row.append(prior[r][c] * (1 - conf) + emp * conf)
            out.append(row)
        return out

    # ---- K-means style prior (layout weighting) ----
    def style_centroid(self):
        if not self.centroids or not self.style:
            return None
        return self.centroids[list(ARCHETYPES).index(self.style)]

    def style_model(self):
        """(centre, variance) per layout feature for the player's K-means style, or None.

        The K-means centroid gives the centre; the games that belong to that cluster give the
        spread. Both are shrunk towards a random player with PRIOR_GAMES pseudo-games, so a
        few games say little and a player with no real habit ends up looking random (weight 1).
        """
        c = self.style_centroid()
        if c is None:
            return None
        feats = [game_features(g["placement"]) for g in self.games]
        j = list(ARCHETYPES).index(self.style)
        members = [f for f in feats if nearest(f, self.centroids) == j] or feats
        k = len(members)
        mu, var_r = random_stats()
        centre = [(k * c[d] + PRIOR_GAMES * mu[d]) / (k + PRIOR_GAMES) for d in range(3)]
        var = [(sum((f[d] - c[d]) ** 2 for f in members) + PRIOR_GAMES * var_r[d]) / (k + PRIOR_GAMES)
               for d in range(3)]
        return centre, var

    def style_evidence(self):
        """How different the style model is from a random player, in nats (KL divergence of the
        per-feature Gaussians). Near 0 = no habit; several nats = a clear habit."""
        m = self.style_model()
        if m is None:
            return 0.0
        (centre, var), (mu, var_r) = m, random_stats()
        return sum(0.5 * math.log(var_r[d] / var[d]) + (var[d] + (centre[d] - mu[d]) ** 2) / (2 * var_r[d]) - 0.5
                   for d in range(3))

    @property
    def style_name(self):
        """K-means style for display; says so honestly when the player shows no clear habit."""
        if self.style is None:
            return None
        return self.style if self.style_evidence() >= STYLE_MIN_EVIDENCE else "No clear habit"

    def kmeans_weight_fn(self):
        """Returns log_weight_fn(masks) for probability.estimate, or None before any games.

        It returns the LOG of the layout weight, already scaled by how strong the evidence for
        the style is (weak evidence -> flat weights). probability.estimate then tempers it
        further if the weights would leave too few layouts that really count."""
        m = self.style_model()
        if m is None:
            return None
        centre, var = m
        power = min(1.0, self.style_evidence() / STYLE_FULL_EVIDENCE)
        return lambda masks: power * style_log_weight(masks, centre, var)

    # ---- replay detector (instance-based learning) ----
    def replay_prior(self):
        """Layouts used in 2+ games, as {'m': belief mass, 'fleets': [(dict name->mask, share)]}.
        None when the player has not repeated a layout."""
        g = len(self.games)
        if g < 2:
            return None
        from collections import Counter
        sigs = Counter()
        for game in self.games:
            sig = tuple(sorted((name, mask_of(ship_cells(length, r, c, bool(h))))
                               for name, length, r, c, h in game["placement"]))
            sigs[sig] += 1
        repeated = {s: n for s, n in sigs.items() if n >= 2}
        if not repeated:
            return None
        rep_games = sum(repeated.values()) - len(repeated)     # games that copied an earlier one
        rate = rep_games / (g - 1)
        m = min(0.85, self.confidence * rate)
        total = sum(repeated.values())
        return {"m": m, "rate": rate, "fleets": [(dict(s), n / total) for s, n in repeated.items()]}

    # ---- evaluation of the decision tree on a held-out game ----
    def holdout(self):
        """Train on all games but the last, then ask: of the 17 cells the tree rates most
        likely, how many really held a ship in the last game? (chance = 17*17/100)."""
        if len(self.games) < 3:
            return None
        train, test = self.games[:-1], self.games[-1]
        rows, labels = [], []
        for g in train:
            occ = set()
            for _, length, r, c, h in g["placement"]:
                occ.update(ship_cells(length, r, c, bool(h)))
            for r in range(SIZE):
                for c in range(SIZE):
                    rows.append(cell_features(r, c))
                    labels.append(1 if (r, c) in occ else 0)
        tree = DecisionTree().fit(rows, labels)
        truth = set()
        for _, length, r, c, h in test["placement"]:
            truth.update(ship_cells(length, r, c, bool(h)))
        groups = {}
        for r in range(SIZE):
            for c in range(SIZE):
                groups.setdefault(round(tree.predict(cell_features(r, c)), 9), []).append((r, c))
        left, hits = TOTAL_SHIP_CELLS, 0.0
        for score in sorted(groups, reverse=True):
            cells = groups[score]
            take = min(left, len(cells))
            hits += take * sum(1 for x in cells if x in truth) / len(cells)
            left -= take
            if left <= 0:
                break
        return {"hits": hits, "of": TOTAL_SHIP_CELLS,
                "chance": TOTAL_SHIP_CELLS * TOTAL_SHIP_CELLS / (SIZE * SIZE)}

    def learn_lines(self):
        """Plain-text description of everything the AI has learned (for the UI)."""
        if not self.games:
            return ["No games recorded yet.",
                    "Finish a game: the AI stores your fleet and shots, then retrains."]
        n = len(self.games)
        feats = [game_features(g["placement"]) for g in self.games]
        mean = [sum(f[k] for f in feats) / n for k in range(3)]
        lines = [f"Games learned from: {n}   confidence = n/(n+4) = {self.confidence:.0%}",
                 f"K-means nearest style: {self.style}   (evidence {self.style_evidence():.1f} nats; "
                 f"below {STYLE_MIN_EVIDENCE:.0f} it counts as no clear habit)",
                 f"Your average features: edge {mean[0]:.2f}, touching {mean[1]:.2f}, horizontal {mean[2]:.2f}",
                 "Distance to each K-means centroid (smaller = closer):"]
        for name, cen in zip(ARCHETYPES, self.centroids):
            lines.append(f"   {name:<17} {_dist(mean, cen) ** 0.5:.3f}")
        sm = self.style_model()
        if sm:
            mu, var_r = random_stats()
            lines.append("Style model used to weight layouts (centre +/- spread; a player with no habits "
                         "would be edge %.2f+/-%.2f, touching %.2f+/-%.2f, horizontal %.2f+/-%.2f):"
                         % (mu[0], var_r[0] ** 0.5, mu[1], var_r[1] ** 0.5, mu[2], var_r[2] ** 0.5))
            lines.append("   edge %.2f+/-%.2f, touching %.2f+/-%.2f, horizontal %.2f+/-%.2f"
                         % (sm[0][0], sm[1][0] ** 0.5, sm[0][1], sm[1][1] ** 0.5, sm[0][2], sm[1][2] ** 0.5))
        hold = self.holdout()
        if hold:
            lines.append(f"Tree check on your last game: of its top {hold['of']} cells, "
                         f"{hold['hits']:.1f} held ships (pure chance: {hold['chance']:.1f})")
        rp = self.replay_prior()
        lines.append(f"Replay detector: you repeated a layout in {rp['rate']:.0%} of games "
                     f"-> {rp['m']:.0%} belief mass on it" if rp else
                     "Replay detector: no repeated layouts yet (it stays inactive)")
        lines.append("Decision tree (probability of a ship, by cell features):")
        lines += ["  " + ln for ln in self.tree.describe()[:16]]
        return lines

    def summary(self, you=True):
        """Three short lines for the UI. you=False words them for an AI-vs-AI match."""
        if not self.games:
            return ["No games recorded yet.",
                    "Finish a game and the AI starts learning your habits." if you
                    else "After the first game the AI starts learning its opponent."]
        wins = sum(1 for g in self.games if g.get("winner") == "player")
        return [f"Games learned from: {len(self.games)} ({'you' if you else 'the opponent'} won {wins})",
                f"{'Your' if you else 'Its'} placement style: {self.style_name}",
                f"Model confidence: {self.confidence:.0%}"]
