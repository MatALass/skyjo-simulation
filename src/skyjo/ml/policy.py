"""Stratégie apprise : joue le coup dont l'après-coup a le plus petit score prédit.

C'est une étape d'« itération de politique » approchée : le modèle estime la valeur des situations sous la
politique qui a généré les données (mélange de personnalités + exploration) ; jouer le meilleur coup selon ce
modèle donne une politique au moins aussi bonne, en théorie. On peut répéter : générer des parties avec
cette nouvelle politique, réentraîner, et ainsi de suite.

- Poser une carte : après-coup connu exactement.
- Retourner une carte cachée : on moyenne la prédiction sur toutes les valeurs possibles, pondérées par les
  cartes encore inconnues (information publique).
- Prendre la défausse ou piocher : on compare le meilleur placement de la défausse à l'espérance du meilleur
  coup après une pioche (moyenne sur les cartes qu'on peut piocher).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from skyjo.board import ROWS, Pos
from skyjo.ml.features import Context, Grid, apply_place, apply_reveal, featurize, grid_from_view, make_context
from skyjo.strategies.base import DrawDecision, GameView, Strategy

DEFAULT_MODEL = "models/value.joblib"


_THREAD_LIMIT: Any = None


def _single_thread() -> None:
    """Un seul thread OpenMP par processus.

    Les simulations parallélisent déjà sur les processus ; laisser scikit-learn lancer en plus autant de
    threads que de cœurs dans chaque processus provoque une sursouscription : mesuré 40x plus lent sur
    2 cœurs (53 s contre 1,3 s pour les mêmes parties), bien pire sur une machine à 32 cœurs.
    """
    global _THREAD_LIMIT
    if _THREAD_LIMIT is None:
        from threadpoolctl import threadpool_limits

        _THREAD_LIMIT = threadpool_limits(limits=1, user_api="openmp")


@lru_cache(maxsize=8)
def load_model(path: str) -> dict[str, Any]:
    import joblib

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Modèle introuvable : {p}. Entraînez-le avec `skyjo ml train`.")
    bundle = joblib.load(p)
    _single_thread()  # après le chargement : la bibliothèque OpenMP de scikit-learn est alors chargée
    return bundle


class MLStrategy(Strategy):
    description = "Stratégie apprise : minimise le score de manche prédit par le modèle de valeur."

    def __init__(self, path: str = DEFAULT_MODEL) -> None:
        bundle = load_model(path)
        self.model = bundle["model"]
        self.path = path
        self.name = "ml" if path == DEFAULT_MODEL else f"ml[{Path(path).stem}]"

    # ------------------------------------------------------------------ évaluation par lots
    def _predict(self, rows: list[list[float]]):
        import numpy as np

        return self.model.predict(np.asarray(rows, dtype=np.float32))

    @staticmethod
    def _state(view: GameView) -> tuple[Grid, Context]:
        me = grid_from_view(view.board)
        opps = [grid_from_view(b) for b in view.opponent_boards]
        return me, make_context(opps, view.belief.expected(), view.is_last_turn, view.turn)

    @staticmethod
    def _positions(me: Grid) -> list[Pos]:
        return [(c, r) for c, col in enumerate(me) if col is not None for r in range(ROWS)]

    def _hidden(self, me: Grid) -> list[Pos]:
        return [p for p in self._positions(me) if me[p[0]][p[1]] is None]  # type: ignore[index]

    def _reveal_rows(self, view: GameView, me: Grid, ctx: Context) -> tuple[list[list[float]], list[Pos], list]:
        """Lignes à évaluer pour « retourner chaque carte cachée », une par valeur possible."""
        hidden = self._hidden(me)
        items = list(view.belief.items())
        rows = [featurize(apply_reveal(me, pos, v), ctx) for pos in hidden for v, _ in items]
        return rows, hidden, items

    @staticmethod
    def _reveal_values(preds, hidden: list[Pos], items: list) -> dict[Pos, float]:
        k = len(items)
        return {pos: float(sum(p * preds[i * k + j] for j, (_, p) in enumerate(items))) for i, pos in enumerate(hidden)}

    def _place_rows(self, me: Grid, ctx: Context, card: int) -> tuple[list[list[float]], list[Pos]]:
        positions = self._positions(me)
        return [featurize(apply_place(me, pos, card), ctx) for pos in positions], positions

    # ------------------------------------------------------------------ décisions
    def initial_reveals(self, view: GameView) -> list[Pos]:
        cols = view.board.columns_left()
        return [(c, view.rng.randrange(ROWS)) for c in view.rng.sample(cols, 2)]

    def take_discard(self, view: GameView) -> bool:
        """Un seul appel au modèle : défausse posée partout, chaque pioche possible posée partout, retournements."""
        if view.top_discard is None:
            return False
        me, ctx = self._state(view)
        positions = self._positions(me)
        n = len(positions)
        items = list(view.belief.items())
        rows = [featurize(apply_place(me, pos, view.top_discard), ctx) for pos in positions]
        rows += [featurize(apply_place(me, pos, c), ctx) for c, _ in items for pos in positions]
        rev_rows, hidden, rev_items = self._reveal_rows(view, me, ctx)
        preds = self._predict(rows + rev_rows)
        take = float(preds[:n].min())
        reveal = self._reveal_values(preds[n + n * len(items):], hidden, rev_items)
        best_reveal = min(reveal.values()) if reveal else float("inf")
        draw_ev = sum(p * min(float(preds[n + i * n:n + (i + 1) * n].min()), best_reveal)
                      for i, (_, p) in enumerate(items))
        if self.explain:
            self.last_explanation = {"step": "ml", "take": take < draw_ev, "take_value": take, "draw_ev": draw_ev}
        return take < draw_ev

    def place_discard(self, view: GameView, card: int) -> Pos:
        me, ctx = self._state(view)
        rows, positions = self._place_rows(me, ctx, card)
        preds = self._predict(rows)
        return positions[int(preds.argmin())]

    def handle_drawn(self, view: GameView, card: int) -> DrawDecision:
        me, ctx = self._state(view)
        rows, positions = self._place_rows(me, ctx, card)
        rev_rows, hidden, items = self._reveal_rows(view, me, ctx)
        preds = self._predict(rows + rev_rows)
        n = len(positions)
        best_place = int(preds[:n].argmin())
        if hidden:
            reveal = self._reveal_values(preds[n:], hidden, items)
            best_reveal = min(reveal, key=reveal.get)
            if reveal[best_reveal] < preds[best_place]:
                return DrawDecision(False, best_reveal)
        return DrawDecision(True, positions[best_place])
