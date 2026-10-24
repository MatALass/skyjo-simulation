"""Tournoi : toutes les tables possibles d'un pool de stratégies, pour chaque nombre de joueurs.

Avec un pool de taille >= n, chaque stratégie joue exactement le même nombre de tables
contre chaque combinaison d'adversaires : la comparaison est équitable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement
from typing import Any

from skyjo.progress import Progress
from skyjo.rules import DEFAULT_RULES, Rules
from skyjo.simulate import Chunk, MatchResult, StrategyStats, plan_chunks, run_chunks


@dataclass
class TournamentResult:
    by_players: dict[int, dict[str, StrategyStats]] = field(default_factory=dict)
    matches: list[MatchResult] = field(default_factory=list)

    def ranking(self) -> list[dict[str, Any]]:
        rows = []
        for n, stats in sorted(self.by_players.items()):
            ranked = sorted((s.summary(n) for s in stats.values()), key=lambda r: -r["win_rate"])
            for rank, r in enumerate(ranked, 1):
                rows.append({"rank": rank, **r})
        return rows

    def match_rows(self) -> list[dict[str, Any]]:
        rows = []
        for i, m in enumerate(self.matches):
            table = " vs ".join(m.specs)
            for r in m.summaries():
                rows.append({"match_id": i, "table": table, "stalled_rounds": m.stalled_rounds, **r})
        return rows


def tables_for(pool: list[str], n: int) -> list[tuple[str, ...]]:
    if len(pool) >= n:
        return list(combinations(pool, n))
    return [t for t in combinations_with_replacement(pool, n) if len(set(t)) > 1 or len(pool) == 1]


def plan_tournament(pool: list[str], player_counts: list[int], n_games: int, seed: int | None,
                    rules: Rules = DEFAULT_RULES) -> list[Chunk]:
    chunks: list[Chunk] = []
    group = 0
    for n in player_counts:
        for t, table in enumerate(tables_for(pool, n)):
            table_seed = None if seed is None else hash((seed, n, t)) & 0xFFFFFFFF
            chunks += plan_chunks(group, list(table), n_games, table_seed, rules)
            group += 1
    return chunks


def run_tournament(
    pool: list[str],
    player_counts: list[int],
    n_games: int,
    seed: int | None = None,
    jobs: int = 1,
    rules: Rules = DEFAULT_RULES,
    progress: Progress | None = None,
) -> TournamentResult:
    matches = run_chunks(plan_tournament(pool, player_counts, n_games, seed, rules), jobs, progress)
    result = TournamentResult(matches=matches)
    for m in matches:
        bucket = result.by_players.setdefault(m.n_players, {})
        for name, s in m.stats.items():
            bucket.setdefault(name, StrategyStats(name)).merge(s)
    return result
