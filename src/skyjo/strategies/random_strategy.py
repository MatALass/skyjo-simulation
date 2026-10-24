"""Référence basse : décisions uniformément aléatoires mais légales."""

from __future__ import annotations

from skyjo.board import Pos
from skyjo.strategies.base import DrawDecision, GameView, Strategy


class RandomStrategy(Strategy):
    name = "random"
    description = "Référence basse : joue au hasard (coups légaux)."

    def take_discard(self, view: GameView) -> bool:
        if self.explain:
            self.last_explanation = {"step": "random"}
        return view.rng.random() < 0.5

    def place_discard(self, view: GameView, card: int) -> Pos:
        if self.explain:
            self.last_explanation = {"step": "random"}
        return view.rng.choice(view.board.positions())

    def handle_drawn(self, view: GameView, card: int) -> DrawDecision:
        if self.explain:
            self.last_explanation = {"step": "random"}
        hidden = view.board.hidden_positions()
        if not hidden or view.rng.random() < 0.5:
            return DrawDecision(True, view.rng.choice(view.board.positions()))
        return DrawDecision(False, view.rng.choice(hidden))
