"""Caractéristiques d'un « après-coup » (afterstate) : ma grille juste après mon coup + le contexte public.

Le modèle apprend V(après-coup) ≈ mon score final de manche (doublement compris). Une même fonction sert
à construire le jeu de données (pendant les simulations) et à jouer (en évaluant chaque coup possible) :
c'est la garantie que le modèle voit en jeu exactement ce qu'il a vu à l'entraînement.

Seule l'information publique est utilisée : cartes visibles, nombre de cartes cachées, espérance des
cartes inconnues (paquet - cartes visibles - défausse). Jamais la valeur d'une carte cachée.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from skyjo.board import COLS, ROWS, BoardView, Pos

Grid = list[list[int | None] | None]

FEATURES: tuple[str, ...] = (
    "my_visible", "my_hidden", "my_est", "cols_removed",
    "cols_hidden_0", "cols_hidden_1", "cols_hidden_2", "cols_hidden_3",
    "open_pairs", "open_pair_value", "closed_pairs", "closed_pair_value",
    "max_visible", "n_high", "n_negative",
    "opp_best_est", "lead", "opp_min_hidden", "opp_mean_hidden", "n_players",
    "is_last_turn", "i_finish", "e_unknown", "turn",
)


@dataclass(frozen=True)
class Context:
    """Partie publique qui ne dépend pas de mon coup (calculée une fois par décision)."""

    opp_best_est: float
    opp_min_hidden: int
    opp_mean_hidden: float
    n_players: int
    is_last_turn: bool
    e_unknown: float
    turn: int


def grid_from_view(view: BoardView) -> Grid:
    cols: Grid = [None] * COLS
    for c, r in view.positions():
        if cols[c] is None:
            cols[c] = [None] * ROWS
        cols[c][r] = view.peek((c, r))  # type: ignore[index]
    return cols


def make_context(opponents: Sequence[Grid], e_unknown: float, is_last_turn: bool, turn: int) -> Context:
    ests, hiddens = [], []
    for g in opponents:
        vis = hid = 0
        for col in g:
            if col is None:
                continue
            for v in col:
                if v is None:
                    hid += 1
                else:
                    vis += v
        ests.append(vis + hid * e_unknown)
        hiddens.append(hid)
    return Context(min(ests), min(hiddens), sum(hiddens) / len(hiddens), len(opponents) + 1,
                   is_last_turn, e_unknown, turn)


def _remove_completed(grid: Grid, col: int) -> None:
    c = grid[col]
    if c is not None and c[0] is not None and c[0] == c[1] == c[2]:
        grid[col] = None


def apply_place(grid: Grid, pos: Pos, card: int) -> Grid:
    """Ma grille après avoir posé `card` en `pos` (colonne retirée si elle devient identique)."""
    g = [None if col is None else list(col) for col in grid]
    c, r = pos
    g[c][r] = card  # type: ignore[index]
    _remove_completed(g, c)
    return g


def apply_reveal(grid: Grid, pos: Pos, value: int) -> Grid:
    return apply_place(grid, pos, value)


def featurize(grid: Grid, ctx: Context) -> list[float]:
    visible = hidden = removed = 0
    by_hidden = [0, 0, 0, 0]
    open_pairs = open_value = closed_pairs = closed_value = 0
    max_visible = -2
    n_high = n_neg = 0
    for col in grid:
        if col is None:
            removed += 1
            continue
        h = 0
        vals = []
        for v in col:
            if v is None:
                h += 1
            else:
                visible += v
                vals.append(v)
                if v > max_visible:
                    max_visible = v
                if v >= 9:
                    n_high += 1
                elif v < 0:
                    n_neg += 1
        hidden += h
        by_hidden[h] += 1
        if h == 1 and vals[0] == vals[1]:
            open_pairs += 1
            open_value += vals[0]
        elif h == 0 and len(set(vals)) == 2:  # deux égales + une différente : paire « fermée »
            pv = max(vals, key=vals.count)
            closed_pairs += 1
            closed_value += pv
    my_est = visible + hidden * ctx.e_unknown
    return [
        visible, hidden, my_est, removed,
        *by_hidden,
        open_pairs, open_value, closed_pairs, closed_value,
        max_visible, n_high, n_neg,
        ctx.opp_best_est, ctx.opp_best_est - my_est, ctx.opp_min_hidden, ctx.opp_mean_hidden, ctx.n_players,
        float(ctx.is_last_turn), float(hidden == 0 and not ctx.is_last_turn), ctx.e_unknown, ctx.turn,
    ]
