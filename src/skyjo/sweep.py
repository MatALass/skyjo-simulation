"""Balayage d'un paramètre : une stratégie déclinée en plusieurs variantes face aux mêmes adversaires.

Exemple : quelle valeur de `hidden_max` rend le Prudent le plus fort à 3 joueurs ?
"""

from __future__ import annotations

from typing import Any

from skyjo.progress import Progress
from skyjo.rules import DEFAULT_RULES, Rules
from skyjo.simulate import Chunk, plan_chunks, run_chunks
from skyjo.strategies import make_strategy


def run_sweep(
    base: str,
    param: str,
    values: list[str],
    opponents: list[str],
    n_games: int,
    seed: int | None = None,
    jobs: int = 1,
    rules: Rules = DEFAULT_RULES,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    specs = [f"{base}:{param}={v}" for v in values]
    names = [make_strategy(s).name for s in specs]  # valide les paramètres avant de lancer
    chunks: list[Chunk] = []
    for g, spec in enumerate(specs):  # même graine pour chaque variante : variance réduite
        chunks += plan_chunks(g, [spec, *opponents], n_games, seed, rules)
    results = run_chunks(chunks, jobs, progress)

    rows = []
    for value, name, match in zip(values, names, results, strict=True):
        mine = next(r for r in match.summaries() if r["strategy"] == name)
        rows.append({"param": param, "value": value, "opponents": " ".join(opponents), **mine})
    return rows
