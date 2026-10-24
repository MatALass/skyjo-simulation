"""Simulation de N parties : sièges tirés au hasard à chaque partie, stats agrégées par stratégie."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from skyjo.game import play_game
from skyjo.progress import Progress, execute
from skyjo.rules import DEFAULT_RULES, Rules
from skyjo.strategies import make_strategy


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class StrategyStats:
    name: str
    appearances: int = 0          # nombre de (partie, joueur) joués avec cette stratégie
    wins: float = 0.0             # égalité = victoire partagée
    game_score_sum: float = 0.0
    rounds: int = 0
    round_score_sum: float = 0.0
    round_score_sq: float = 0.0
    round_wins: float = 0.0       # manche gagnée = plus petit score de la manche
    finishes: int = 0
    doubled: int = 0

    def merge(self, other: StrategyStats) -> None:
        for f in ("appearances", "wins", "game_score_sum", "rounds", "round_score_sum",
                  "round_score_sq", "round_wins", "finishes", "doubled"):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    def summary(self, n_players: int) -> dict[str, Any]:
        win_rate = self.wins / self.appearances if self.appearances else 0.0
        lo, hi = wilson_interval(win_rate, self.appearances)
        mean = self.round_score_sum / self.rounds if self.rounds else 0.0
        var = max(self.round_score_sq / self.rounds - mean**2, 0.0) if self.rounds else 0.0
        # IC approximatif : les manches d'une même partie ne sont pas indépendantes.
        ci = 1.96 * math.sqrt(var / self.rounds) if self.rounds else 0.0
        return {
            "n_players": n_players,
            "strategy": self.name,
            "appearances": self.appearances,
            "win_rate": win_rate,
            "win_rate_ci_low": lo,
            "win_rate_ci_high": hi,
            "win_index": win_rate * n_players,  # 1.0 = aussi bon que le hasard
            "avg_game_score": self.game_score_sum / self.appearances if self.appearances else 0.0,
            "rounds": self.rounds,
            "avg_round_score": mean,
            "round_score_std": math.sqrt(var),
            "round_score_ci95": ci,
            "round_win_rate": self.round_wins / self.rounds if self.rounds else 0.0,
            "finish_rate": self.finishes / self.rounds if self.rounds else 0.0,
            "doubled_rate": self.doubled / self.finishes if self.finishes else 0.0,
        }


@dataclass
class MatchResult:
    specs: list[str]
    n_games: int = 0
    stats: dict[str, StrategyStats] = field(default_factory=dict)
    total_rounds: int = 0
    stalled_rounds: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_players(self) -> int:
        return len(self.specs)

    def merge(self, other: MatchResult) -> None:
        self.n_games += other.n_games
        self.total_rounds += other.total_rounds
        self.stalled_rounds += other.stalled_rounds
        self.decisions.extend(other.decisions)
        for name, s in other.stats.items():
            self.stats.setdefault(name, StrategyStats(name)).merge(s)

    def summaries(self) -> list[dict[str, Any]]:
        rows = [s.summary(self.n_players) for s in self.stats.values()]
        return sorted(rows, key=lambda r: -r["win_rate"])


def _run_chunk(specs: list[str], n_games: int, seed: str | None, rules: Rules,
               log_games: int, game_offset: int) -> MatchResult:
    rng = random.Random(seed)
    strategies = [make_strategy(s) for s in specs]
    n = len(strategies)
    result = MatchResult(list(specs), n_games)
    for s in strategies:
        result.stats.setdefault(s.name, StrategyStats(s.name))

    for g in range(n_games):
        order = list(range(n))
        rng.shuffle(order)  # siège -> indice de stratégie
        record = g < log_games
        game = play_game([strategies[i] for i in order], rng, rules, record=record)
        names = [strategies[i].name for i in order]

        for seat, name in enumerate(names):
            st = result.stats[name]
            st.appearances += 1
            st.game_score_sum += game.totals[seat]
        for w in game.winners:
            result.stats[names[w]].wins += 1 / len(game.winners)

        for rnd in game.rounds:
            result.total_rounds += 1
            result.stalled_rounds += rnd.stalled
            best = min(rnd.scores)
            n_best = rnd.scores.count(best)
            for seat, score in enumerate(rnd.scores):
                st = result.stats[names[seat]]
                st.rounds += 1
                st.round_score_sum += score
                st.round_score_sq += score * score
                if score == best:
                    st.round_wins += 1 / n_best
            if rnd.finisher is not None:
                st = result.stats[names[rnd.finisher]]
                st.finishes += 1
                st.doubled += rnd.finisher_doubled

        if record:
            for d in game.decisions:
                d["game"] = game_offset + g
            result.decisions.extend(game.decisions)
    return result


CHUNK_GAMES = 100  # taille fixe des paquets : résultats identiques quel que soit --jobs


@dataclass(frozen=True)
class Chunk:
    """Paquet de parties d'une même table, unité de travail parallèle."""

    group: int                # table / variante à laquelle appartient le paquet
    specs: tuple[str, ...]
    n_games: int
    seed: str | None
    rules: Rules
    log_games: int
    game_offset: int


def plan_chunks(group: int, specs: list[str], n_games: int, seed: int | None, rules: Rules,
                log_games: int = 0, chunk_games: int = CHUNK_GAMES) -> list[Chunk]:
    chunks = []
    for i, offset in enumerate(range(0, n_games, chunk_games)):
        size = min(chunk_games, n_games - offset)
        chunks.append(Chunk(
            group=group,
            specs=tuple(specs),
            n_games=size,
            seed=None if seed is None else f"{seed}:{i}",
            rules=rules,
            log_games=max(0, min(size, log_games - offset)),
            game_offset=offset,
        ))
    return chunks


def _run_chunk_task(chunk: Chunk) -> MatchResult:
    return _run_chunk(list(chunk.specs), chunk.n_games, chunk.seed, chunk.rules,
                      chunk.log_games, chunk.game_offset)


def run_chunks(chunks: list[Chunk], jobs: int = 1, progress: Progress | None = None) -> list[MatchResult]:
    """Exécute les paquets et fusionne par table ; un MatchResult par groupe, trié par numéro de groupe."""
    parts = execute(chunks, _run_chunk_task, jobs, progress,
                    size=lambda c: c.n_games, group=lambda c: c.group)
    merged: dict[int, MatchResult] = {}
    for chunk, part in zip(chunks, parts, strict=True):
        merged.setdefault(chunk.group, MatchResult(list(chunk.specs))).merge(part)
    return [merged[g] for g in sorted(merged)]


def run_match(
    specs: list[str],
    n_games: int,
    seed: int | None = None,
    jobs: int = 1,
    log_games: int = 0,
    rules: Rules = DEFAULT_RULES,
    progress: Progress | None = None,
) -> MatchResult:
    """`log_games` : nombre de parties (les premières) dont chaque décision est enregistrée."""
    return run_chunks(plan_chunks(0, specs, n_games, seed, rules, log_games), jobs, progress)[0]
