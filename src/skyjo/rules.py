"""Paramètres de règles. Valeurs par défaut = livret officiel ; les variantes sont configurables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    end_score: int = 100
    # Colonne de 3 cartes identiques visibles retirée...
    # ...aussi quand elle est complétée par un retournement (pas seulement par un échange)
    columns_on_reveal: bool = True
    columns_at_final_reveal: bool = True  # ...aussi lors du retournement final (règle officielle)
    # Garde-fou : au-delà, la manche est arrêtée et comptée "bloquée" (sans doublement).
    max_turns_per_round: int = 600
    max_rounds_per_game: int = 200


DEFAULT_RULES = Rules()
