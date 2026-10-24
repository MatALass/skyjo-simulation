"""Registre des stratégies.

Spécification texte : "nom" ou "nom:param=valeur:param=valeur"
ex. "prudent", "stratege:finish_margin=0", "fonceur:hidden_max=5:reveal_bonus=1.5"
"""

from __future__ import annotations

from skyjo.strategies.base import DrawDecision, GameView, Strategy
from skyjo.strategies.heuristic import HeuristicStrategy, Profile
from skyjo.strategies.personalities import LAMBDA_PLAYERS, PROFILES
from skyjo.strategies.random_strategy import RandomStrategy

__all__ = [
    "DrawDecision",
    "GameView",
    "HeuristicStrategy",
    "LAMBDA_PLAYERS",
    "PROFILES",
    "Profile",
    "RandomStrategy",
    "Strategy",
    "available",
    "make_strategy",
]


def available() -> dict[str, str]:
    """Nom -> description de toutes les stratégies disponibles."""
    out = {name: p.description for name, p in PROFILES.items()}
    out[RandomStrategy.name] = RandomStrategy.description
    return out


def make_strategy(spec: str) -> Strategy:
    name, *params = [s.strip() for s in spec.split(":")]
    if name == RandomStrategy.name:
        if params:
            raise ValueError("La stratégie 'random' n'a pas de paramètres.")
        return RandomStrategy()
    if name not in PROFILES:
        raise ValueError(f"Stratégie inconnue '{name}'. Disponibles : {', '.join(available())}")
    profile = PROFILES[name]
    if params:
        overrides = {}
        for item in params:
            if "=" not in item:
                raise ValueError(f"Paramètre mal formé '{item}' (attendu clé=valeur).")
            k, v = item.split("=", 1)
            overrides[k.strip()] = v.strip()
        profile = profile.with_overrides(overrides)
    return HeuristicStrategy(profile)
