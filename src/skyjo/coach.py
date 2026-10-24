"""Coach : quel est le meilleur coup dans une situation précise ?

Méthode (Monte Carlo à nombres aléatoires communs) :
1. On tire les cartes inconnues (cartes cachées de tous les plateaux + pioche) parmi celles qui restent.
2. Pour ce même tirage, on joue chaque coup possible puis on simule la fin de la manche
   (tout le monde joue ensuite avec une stratégie donnée, `expert` par défaut).
3. On répète des centaines de fois : la différence moyenne entre deux coups mesure *l'effet du coup*,
   pas une corrélation. Les coups nettement moins bons sont éliminés en cours de route.

Limites : on évalue la manche, pas la partie (le score cumulé est ignoré). Les cartes de la défausse
autres que celle du dessus sont supposées inconnues, sauf si on les fournit (`seen`).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from skyjo.board import COLS, ROWS, Board, Pos, Slot
from skyjo.cards import CARD_DISTRIBUTION
from skyjo.game import Round
from skyjo.progress import Progress, execute
from skyjo.rules import DEFAULT_RULES, Rules
from skyjo.strategies import make_strategy

# Une grille : liste de 4 colonnes ; colonne = 3 valeurs (None = carte cachée) ou None (colonne retirée).
Grid = list[list[int | None] | None]

DEFAULT_POLICY = "expert"
_PRUNE_Z = 2.58


# ---------------------------------------------------------------------------------------------- saisie
def parse_grid(text: str) -> Grid:
    """'5 ? ? 3 / ? 12 ? x / 0 ? ? x' : 3 lignes séparées par '/', 4 cartes par ligne.

    '?' = carte cachée, 'x' = colonne retirée (toute la colonne), sinon la valeur visible.
    """
    rows = [r.split() for r in text.strip().split("/")]
    if len(rows) != ROWS or any(len(r) != COLS for r in rows):
        raise ValueError(f"Grille invalide '{text}' : {ROWS} lignes de {COLS} cartes séparées par '/'.")
    grid: Grid = []
    for c in range(COLS):
        tokens = [rows[r][c].lower() for r in range(ROWS)]
        if all(t == "x" for t in tokens):
            grid.append(None)
            continue
        if "x" in tokens:
            raise ValueError(f"Colonne {c + 1} : 'x' doit couvrir toute la colonne.")
        col: list[int | None] = []
        for t in tokens:
            if t == "?":
                col.append(None)
            else:
                v = int(t)
                if v not in CARD_DISTRIBUTION:
                    raise ValueError(f"Valeur impossible : {v}")
                col.append(v)
        grid.append(col)
    return grid


def format_grid(grid: Grid) -> str:
    lines = []
    for r in range(ROWS):
        cells = []
        for col in grid:
            if col is None:
                cells.append("  x")
            else:
                v = col[r]
                cells.append("  ?" if v is None else f"{v:>3}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def _hidden(grid: Grid) -> list[Pos]:
    return [(c, r) for c, col in enumerate(grid) if col is not None for r in range(ROWS) if col[r] is None]


def _visible_values(grid: Grid) -> list[int]:
    return [v for col in grid if col is not None for v in col if v is not None]


@dataclass(frozen=True)
class Situation:
    me: Grid
    opponents: tuple[Grid, ...]       # dans l'ordre de jeu : [0] joue juste après moi
    top_discard: int | None
    drawn: int | None = None          # carte déjà piochée (sinon : décision de début de tour)
    seen: tuple[int, ...] = ()        # autres cartes connues hors jeu (fond de défausse, colonnes retirées)

    def __post_init__(self) -> None:
        if not 1 <= len(self.opponents) <= 7:
            raise ValueError("Il faut 1 à 7 adversaires.")
        if not _hidden(self.me):
            raise ValueError("Vous n'avez plus de carte cachée : la manche est déjà finie pour vous.")
        finished = [i for i, g in enumerate(self.opponents) if not _hidden(g)]
        if len(finished) > 1:
            raise ValueError("Un seul adversaire peut avoir tout retourné.")
        if self.top_discard is None and self.drawn is None:
            raise ValueError("Indiquez la carte du dessus de la défausse (ou la carte piochée).")
        unknown = self.unknown_counts()
        if any(v < 0 for v in unknown.values()):
            raise ValueError(f"Trop de cartes d'une même valeur : {[k for k, v in unknown.items() if v < 0]}")
        n_hidden = len(_hidden(self.me)) + sum(len(_hidden(g)) for g in self.opponents)
        if sum(unknown.values()) < n_hidden:
            raise ValueError("Pas assez de cartes inconnues pour remplir les cartes cachées.")

    @property
    def finisher(self) -> int | None:
        """Siège (1 = premier adversaire) de celui qui a déjà tout retourné, s'il existe."""
        for i, g in enumerate(self.opponents):
            if not _hidden(g):
                return i + 1
        return None

    def unknown_counts(self) -> dict[int, int]:
        counts = Counter(CARD_DISTRIBUTION)
        known = _visible_values(self.me) + [v for g in self.opponents for v in _visible_values(g)]
        known += list(self.seen)
        if self.top_discard is not None:
            known.append(self.top_discard)
        if self.drawn is not None:
            known.append(self.drawn)
        counts.subtract(known)
        return dict(counts)


# ---------------------------------------------------------------------------------------------- coups
@dataclass(frozen=True)
class Move:
    kind: str            # "take_discard" | "draw" | "replace" | "reveal"
    pos: Pos | None = None

    def label(self, s: Situation) -> str:
        if self.kind == "draw":
            return "Piocher (puis jouer au mieux)"
        assert self.pos is not None
        c, r = self.pos
        col = s.me[c]
        assert col is not None
        target = "cachée" if col[r] is None else f"un {col[r]}"
        where = f"col {c + 1}, ligne {r + 1} ({target})"
        if self.kind == "take_discard":
            return f"Prendre le {s.top_discard} de la défausse → {where}"
        if self.kind == "replace":
            return f"Poser le {s.drawn} → {where}"
        return f"Défausser le {s.drawn}, retourner {where}"


def legal_moves(s: Situation) -> list[Move]:
    positions = [(c, r) for c, col in enumerate(s.me) if col is not None for r in range(ROWS)]
    if s.drawn is None:
        moves = [Move("take_discard", p) for p in positions] if s.top_discard is not None else []
        return moves + [Move("draw")]
    return [Move("replace", p) for p in positions] + [Move("reveal", p) for p in _hidden(s.me)]


# ---------------------------------------------------------------------------------------------- simulation
def _deal(s: Situation, rng: random.Random) -> tuple[list[Board], list[int]]:
    """Tire les cartes cachées et la pioche, compatibles avec ce qui est visible."""
    pool = [v for v, n in s.unknown_counts().items() for _ in range(n)]
    rng.shuffle(pool)
    boards = []
    for grid in (s.me, *s.opponents):
        columns: list[list[Slot] | None] = []
        for col in grid:
            if col is None:
                columns.append(None)
            else:
                columns.append([Slot(v, True) if v is not None else Slot(pool.pop(), False) for v in col])
        boards.append(Board.from_columns(columns))
    return boards, pool  # le reste du paquet forme la pioche


@dataclass(frozen=True)
class RolloutTask:
    situation: Situation
    moves: tuple[Move, ...]
    samples: tuple[int, ...]
    seed: int
    policy: str
    opp_policy: str
    rules: Rules


def _play_move(rnd: Round, s: Situation, move: Move, finisher: int | None) -> None:
    """Joue mon coup (siège 0) sur la manche reconstituée."""
    if move.kind == "take_discard":
        card = rnd.piles.take_discard()
        assert move.pos is not None
        rnd.apply_action(0, "discard", card, move.pos, replace=True)
    elif move.kind == "draw":
        card = rnd.piles.draw()
        decision = rnd.strategies[0].handle_drawn(rnd._view(0, finisher is not None, 0, in_hand=card), card)
        rnd.apply_action(0, "draw", card, decision.pos, decision.replace)
    else:
        assert s.drawn is not None and move.pos is not None
        rnd.apply_action(0, "draw", s.drawn, move.pos, replace=move.kind == "replace")


def _rollouts(task: RolloutTask) -> dict[Move, list[tuple[int, bool, bool, bool]]]:
    s = task.situation
    n = 1 + len(s.opponents)
    strategies = [make_strategy(task.policy)] + [make_strategy(task.opp_policy) for _ in s.opponents]
    discard = [s.top_discard] if s.top_discard is not None else []
    out: dict[Move, list[tuple[int, bool, bool, bool]]] = {m: [] for m in task.moves}

    for k in task.samples:
        boards, draw_pile = _deal(s, random.Random(f"{task.seed}:deal:{k}"))
        for move in task.moves:
            rng = random.Random(f"{task.seed}:play:{k}")  # même suite aléatoire pour chaque coup
            rnd = Round.from_position(strategies, rng, [b.copy() for b in boards], draw_pile, discard, task.rules)
            finisher = s.finisher
            _play_move(rnd, s, move, finisher)
            if finisher is None:
                if rnd.boards[0].all_visible():
                    finisher, remaining = 0, n - 1
                else:
                    remaining = 0
            else:
                remaining = finisher - 1  # adversaires entre moi et le finisseur
            res = rnd.continue_from(1 % n, finisher, remaining, turns=1)
            score = res.scores[0]
            out[move].append((score, score == min(res.scores), res.finisher == 0 and res.finisher_doubled,
                              res.finisher == 0))
    return out


# ---------------------------------------------------------------------------------------------- résultat
@dataclass
class MoveStats:
    move: Move
    label: str
    scores: list[int] = field(default_factory=list)
    won: int = 0
    doubled: int = 0
    finished: int = 0
    eliminated_at: int | None = None
    diff_mean: float = 0.0
    diff_ci: float = 0.0

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return sum(self.scores) / self.n if self.n else float("nan")


def _paired_diff(a: list[int], b: list[int]) -> tuple[float, float]:
    """Moyenne et IC 95 % de a - b sur les tirages communs."""
    m = min(len(a), len(b))
    d = [a[i] - b[i] for i in range(m)]
    if m < 2:
        return (d[0] if d else 0.0), float("inf")
    mean = sum(d) / m
    var = sum((x - mean) ** 2 for x in d) / (m - 1)
    return mean, 1.96 * math.sqrt(var / m)


def coach(
    s: Situation,
    samples: int = 400,
    batch: int = 50,
    seed: int = 0,
    policy: str = DEFAULT_POLICY,
    opp_policy: str | None = None,
    jobs: int = 1,
    rules: Rules = DEFAULT_RULES,
    progress: Progress | None = None,
    prune: bool = True,
) -> list[MoveStats]:
    """Évalue chaque coup ; renvoie les coups triés du meilleur au moins bon."""
    opp_policy = opp_policy or policy
    make_strategy(policy), make_strategy(opp_policy)  # valide avant de lancer
    stats = {m: MoveStats(m, m.label(s)) for m in legal_moves(s)}
    active = list(stats)
    done = 0
    while done < samples and active:
        size = min(batch, samples - done)
        ids = list(range(done, done + size))
        per_task = max(1, math.ceil(size / max(jobs, 1)))
        tasks = [RolloutTask(s, tuple(active), tuple(ids[i:i + per_task]), seed, policy, opp_policy, rules)
                 for i in range(0, size, per_task)]
        for part in execute(tasks, _rollouts, jobs, progress, size=lambda t: len(t.samples)):
            for move, results in part.items():
                st = stats[move]
                for score, won, doubled, finished in results:
                    st.scores.append(score)
                    st.won += won
                    st.doubled += doubled
                    st.finished += finished
        done += size

        best = min(active, key=lambda m: stats[m].mean)
        for m in active:
            stats[m].diff_mean, stats[m].diff_ci = _paired_diff(stats[m].scores, stats[best].scores)
        if prune and done < samples:
            # Élimine les coups nettement moins bons que le meilleur. Seuil à 99 % (et non 95 %) :
            # on compare une vingtaine de coups à chaque étape, il faut limiter les faux rejets.
            for m in list(active):
                st = stats[m]
                if m is not best and done >= 2 * batch and st.diff_mean - st.diff_ci * _PRUNE_Z / 1.96 > 0:
                    st.eliminated_at = done
                    active.remove(m)

    best_move = min(stats.values(), key=lambda st: (st.eliminated_at is not None, st.mean)).move
    for st in stats.values():
        st.diff_mean, st.diff_ci = _paired_diff(st.scores, stats[best_move].scores)
    return sorted(stats.values(), key=lambda st: (st.eliminated_at is not None, st.mean))


def format_report(s: Situation, results: Sequence[MoveStats]) -> str:
    lines = ["Mon plateau :", format_grid(s.me), ""]
    for i, g in enumerate(s.opponents, 1):
        lines += [f"Adversaire {i} :", format_grid(g), ""]
    ctx = f"Défausse : {s.top_discard}"
    if s.drawn is not None:
        ctx += f" | carte piochée : {s.drawn}"
    if s.finisher is not None:
        ctx += f" | dernier tour (adversaire {s.finisher} a tout retourné)"
    lines += [ctx, ""]
    header = f"{'#':>2}  {'coup':<52}{'score moyen':>12}{'écart vs meilleur':>22}{'manche gagnée':>15}" \
             f"{'je finis':>10}{'doublé':>8}{'simul.':>8}"
    lines += [header, "-" * len(header)]
    for rank, st in enumerate(results, 1):
        diff = "meilleur" if rank == 1 else f"+{st.diff_mean:.2f} ± {st.diff_ci:.2f}"
        mark = "" if st.eliminated_at is None else " (écarté)"
        lines.append(
            f"{rank:>2}  {(st.label + mark)[:52]:<52}{st.mean:>12.2f}{diff:>22}{st.won / st.n:>15.1%}"
            f"{st.finished / st.n:>10.1%}{st.doubled / st.n:>8.1%}{st.n:>8}"
        )
    lines += ["", "Score moyen = mon score de manche attendu (doublement inclus). Plus bas = mieux.",
              "Écart vs meilleur : différence sur les mêmes tirages ; si l'IC contient 0, les deux coups se valent."]
    return "\n".join(lines)
