"""Interface commune des stratégies."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from skyjo.board import BoardView, Pos
from skyjo.cards import CardBelief


@dataclass(frozen=True)
class GameView:
    """Ce qu'un joueur peut observer au moment de décider (information publique + son plateau)."""

    board: BoardView
    opponent_boards: tuple[BoardView, ...]  # dans l'ordre de jeu : [0] = joueur suivant
    top_discard: int | None
    belief: CardBelief                      # cartes inconnues (comptage parfait de l'information publique)
    is_last_turn: bool                      # un adversaire a déjà tout retourné
    turn: int                               # nombre de tours joués dans la manche
    rng: random.Random


@dataclass(frozen=True)
class DrawDecision:
    """Après avoir pioché : échanger (replace=True) ou défausser et retourner `pos`."""

    replace: bool
    pos: Pos


class Strategy(ABC):
    name: str = "abstract"
    description: str = ""
    # Mode replay : la stratégie décrit son raisonnement dans `last_explanation` (dict) après chaque décision.
    # Ne doit jamais changer la décision ni consommer de tirage aléatoire supplémentaire.
    explain: bool = False
    last_explanation: dict | None = None

    def initial_reveals(self, view: GameView) -> list[Pos]:
        """Deux cartes à retourner en début de manche (par défaut : au hasard)."""
        return view.rng.sample(view.board.hidden_positions(), 2)

    @abstractmethod
    def take_discard(self, view: GameView) -> bool:
        """True = prendre la carte de la défausse, False = piocher."""

    @abstractmethod
    def place_discard(self, view: GameView, card: int) -> Pos:
        """Où poser la carte prise dans la défausse (échange obligatoire)."""

    @abstractmethod
    def handle_drawn(self, view: GameView, card: int) -> DrawDecision:
        """Que faire de la carte piochée."""
