"""Génération du jeu de données : une ligne par coup joué, vue par le joueur qui l'a joué.

X = caractéristiques de l'après-coup (features.py) ; y = score final de manche de ce joueur (doublement compris).
Les tables mélangent des personnalités et chaque joueur joue un coup au hasard avec la probabilité `eps`
(exploration) : le modèle voit ainsi aussi des situations que les bons joueurs évitent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skyjo.board import Pos
from skyjo.game import Round, play_game
from skyjo.ml.features import FEATURES, featurize, grid_from_view, make_context
from skyjo.progress import Progress, execute
from skyjo.rules import DEFAULT_RULES, Rules
from skyjo.strategies import make_strategy
from skyjo.strategies.base import DrawDecision, GameView, Strategy

DEFAULT_POOL = ("expert", "stratege", "prudent", "collectionneur", "opportuniste", "fonceur")
META = ("game", "round", "seat", "strategy", "explored")
TARGETS = ("round_score", "raw_round_score", "won_round")


class Explorer(Strategy):
    """Joue comme `base`, sauf avec la probabilité `eps` où il joue un coup légal au hasard."""

    def __init__(self, base: Strategy, eps: float) -> None:
        self.base, self.eps = base, eps
        self.name = base.name
        self.explored = False

    def _roll(self, view: GameView) -> bool:
        self.explored = view.rng.random() < self.eps
        return self.explored

    def initial_reveals(self, view: GameView) -> list[Pos]:
        return self.base.initial_reveals(view)

    def take_discard(self, view: GameView) -> bool:
        if self._roll(view):
            return view.rng.random() < 0.5
        return self.base.take_discard(view)

    def place_discard(self, view: GameView, card: int) -> Pos:
        if self.explored:
            return view.rng.choice(view.board.positions())
        return self.base.place_discard(view, card)

    def handle_drawn(self, view: GameView, card: int) -> DrawDecision:
        if self._roll(view):
            hidden = view.board.hidden_positions()
            if not hidden or view.rng.random() < 0.5:
                return DrawDecision(True, view.rng.choice(view.board.positions()))
            return DrawDecision(False, view.rng.choice(hidden))
        return self.base.handle_drawn(view, card)


class Recorder:
    """Observateur : enregistre l'après-coup de chaque tour, puis la cible à la fin de la manche."""

    def __init__(self, game_id: int, names: list[str]) -> None:
        self.game_id, self.names = game_id, names
        self.round = 0
        self.rows: list[list[Any]] = []
        self._pending: list[tuple[int, list[Any]]] = []

    def __call__(self, event: str, rnd: Round, payload: dict[str, Any]) -> None:
        if event == "deal":
            self.round += 1
        elif event == "turn":
            i = payload["player"]
            n = len(rnd.boards)
            me = grid_from_view(rnd.views[i])
            opps = [grid_from_view(rnd.views[(i + k) % n]) for k in range(1, n)]
            ctx = make_context(opps, rnd._belief().expected(), payload["is_last_turn"], payload["turn"])
            strat = rnd.strategies[i]
            meta = [self.game_id, self.round, i, self.names[i], int(getattr(strat, "explored", False))]
            self._pending.append((i, meta + featurize(me, ctx)))
        elif event == "round_end":
            res = payload["result"]
            best = min(res.scores)
            for i, row in self._pending:
                self.rows.append(row + [res.scores[i], res.raw_scores[i], int(res.scores[i] == best)])
            self._pending = []


@dataclass(frozen=True)
class GenTask:
    first_game: int
    n_games: int
    seed: str
    pool: tuple[str, ...]
    eps: float
    sizes: tuple[int, ...]
    rules: Rules


def _generate_chunk(task: GenTask) -> list[list[Any]]:
    rng = random.Random(task.seed)
    rows: list[list[Any]] = []
    for g in range(task.n_games):
        n = rng.choice(task.sizes)
        specs = [rng.choice(task.pool) for _ in range(n)]
        strategies: list[Strategy] = [Explorer(make_strategy(s), task.eps) for s in specs]
        rec = Recorder(task.first_game + g, [s.name for s in strategies])
        play_game(strategies, rng, task.rules, observer=rec)
        rows.extend(rec.rows)
    return rows


def generate(n_games: int, seed: int = 0, jobs: int = 1, pool: tuple[str, ...] = DEFAULT_POOL, eps: float = 0.1,
             sizes: tuple[int, ...] = (2, 3, 4), rules: Rules = DEFAULT_RULES, chunk: int = 50,
             progress: Progress | None = None):
    """Renvoie un DataFrame pandas (une ligne par coup)."""
    import pandas as pd

    for s in pool:
        make_strategy(s)  # valide les specs avant de lancer les processus
    tasks = [GenTask(start, min(chunk, n_games - start), f"{seed}:{start}", tuple(pool), eps, tuple(sizes), rules)
             for start in range(0, n_games, chunk)]
    parts = execute(tasks, _generate_chunk, jobs, progress, size=lambda t: t.n_games)
    columns = [*META, *FEATURES, *TARGETS]
    return pd.DataFrame([r for part in parts for r in part], columns=columns)


def save(df, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load(path: str | Path):
    import pandas as pd

    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
