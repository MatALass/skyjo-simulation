"""Optimisation d'un profil par descente de coordonnées.

À chaque passe, chaque paramètre est balayé à tour de rôle (les autres restant fixés) ; on garde la
meilleure valeur si elle améliore l'objectif d'au moins `min_gain`. On s'arrête quand une passe complète
ne change plus rien.

Objectif = moyenne de l'indice de victoire (taux × nb joueurs, 1.0 = hasard) sur plusieurs "terrains"
(tables d'adversaires). Optimiser contre une seule table donne un réglage qui exploite *ces* adversaires ;
plusieurs terrains donnent un réglage robuste.

Garde-fou contre le sur-apprentissage : toutes les évaluations d'une recherche utilisent la même graine
(comparaisons à variance réduite), puis le profil final est revalidé sur une graine neuve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from skyjo.progress import Progress
from skyjo.rules import DEFAULT_RULES, Rules
from skyjo.simulate import Chunk, plan_chunks, run_chunks
from skyjo.strategies import LAMBDA_PLAYERS, HeuristicStrategy, make_strategy
from skyjo.strategies.heuristic import Profile, _parse_value

# Valeurs testées par paramètre. Les booléens d'information (comptage, lookahead) sont exclus :
# ils définissent la personnalité plus qu'un réglage.
DEFAULT_GRID: dict[str, list[str]] = {
    "hidden_max": ["2", "3", "4", "5", "none"],
    "discard_min_gain": ["2", "3", "4", "5"],
    "reveal_bonus": ["-1", "0", "0.5", "1"],
    "tempo_weight": ["0", "1"],
    "pair_bonus": ["0", "1", "2", "4"],
    "pair_scale": ["0", "0.25", "0.5", "1"],
    "pair_min": ["-2", "0", "1"],
    "reveal_spread": ["0", "1", "2"],
    "deny_weight": ["0", "0.5", "1"],
    "hidden_discount": ["0", "1", "2"],
    "finish_margin": ["0", "2", "5", "10"],
    "finish_bonus": ["0", "1", "3", "6"],
    "opp_last_turn_gain": ["1", "3", "5"],
}

DEFAULT_FIELDS: list[list[str]] = [
    ["prudent", "collectionneur", "opportuniste"],  # 4 joueurs, table "sérieuse"
    ["prudent", "fonceur", "collectionneur"],       # 4 joueurs, avec un joueur pressé
    ["prudent"],                                    # 2 joueurs
]


def fields_for_players(player_counts: list[int], pool: tuple[str, ...] = LAMBDA_PLAYERS) -> list[list[str]]:
    """Toutes les tables de joueurs lambda pour chaque taille demandée (le profil optimisé occupe 1 siège)."""
    fields = []
    for n in player_counts:
        if not 2 <= n <= len(pool) + 1:
            raise ValueError(f"Nombre de joueurs {n} : entre 2 et {len(pool) + 1} avec ce pool.")
        fields += [list(c) for c in combinations(pool, n - 1)]
    return fields


def relevant_params(profile: Profile) -> list[str]:
    """Paramètres qui ont un effet pour ce profil."""
    params = [p for p in DEFAULT_GRID if p not in ("finish_margin", "finish_bonus", "opp_last_turn_gain")]
    if profile.draw_lookahead:
        params.remove("discard_min_gain")  # ignoré quand on compare à l'espérance de la pioche
    if profile.finish_policy == "if_lowest":
        params += ["finish_margin", "finish_bonus"]
    elif profile.finish_policy == "probabilistic":
        params += ["finish_bonus", "opp_last_turn_gain"]
    return params


def split_spec(spec: str) -> tuple[str, dict[str, str]]:
    name, *params = [x.strip() for x in spec.split(":")]
    return name, dict(item.split("=", 1) for item in params)


def build_spec(name: str, overrides: dict[str, str]) -> str:
    return ":".join([name, *(f"{k}={v}" for k, v in sorted(overrides.items()))])


@dataclass
class Evaluation:
    spec: str
    score: float                      # objectif : indice de victoire moyen sur les terrains
    per_field: list[dict[str, Any]]   # résumé de la stratégie testée, terrain par terrain


@dataclass
class OptimizeResult:
    base_spec: str
    best_spec: str
    history: list[dict[str, Any]] = field(default_factory=list)   # chaque évaluation
    steps: list[str] = field(default_factory=list)                # changements retenus
    validation: list[Evaluation] = field(default_factory=list)    # base puis meilleur, graine neuve


class Optimizer:
    def __init__(
        self,
        base: str,
        fields: list[list[str]],
        n_games: int,
        seed: int,
        jobs: int = 1,
        rules: Rules = DEFAULT_RULES,
        progress_factory: Callable[[int, str, int], Progress | None] | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.name, self.start = split_spec(base)
        strat = make_strategy(base)
        if not isinstance(strat, HeuristicStrategy):
            raise ValueError("Seuls les profils heuristiques sont optimisables.")
        self.fields = fields
        self.n_games = n_games
        self.seed = seed
        self.jobs = jobs
        self.rules = rules
        self.progress_factory = progress_factory or (lambda total, label, groups: None)
        self.log = log
        self._cache: dict[tuple[int, str], Evaluation] = {}
        self.history: list[dict[str, Any]] = []

    def _profile(self, overrides: dict[str, str]) -> Profile:
        strat = make_strategy(build_spec(self.name, overrides))
        assert isinstance(strat, HeuristicStrategy)
        return strat.p

    def evaluate(self, configs: list[dict[str, str]], seed: int, label: str, n_games: int | None = None,
                 record: bool = True) -> list[Evaluation]:
        n_games = n_games or self.n_games
        specs = [build_spec(self.name, c) for c in configs]
        todo = [s for s in dict.fromkeys(specs) if (seed, s) not in self._cache]
        if todo:
            chunks: list[Chunk] = []
            groups = []
            for spec in todo:
                for f in self.fields:
                    groups.append((spec, len(f) + 1))
                    chunks += plan_chunks(len(groups) - 1, [spec, *f], n_games, seed, self.rules)
            pr = self.progress_factory(len(groups) * n_games, label, len(groups))
            if pr is not None:
                with pr:
                    matches = run_chunks(chunks, self.jobs, pr)
            else:
                matches = run_chunks(chunks, self.jobs)
            for i, spec in enumerate(todo):
                per_field = []
                for j in range(len(self.fields)):
                    m = matches[i * len(self.fields) + j]
                    name = make_strategy(spec).name
                    per_field.append(next(r for r in m.summaries() if r["strategy"] == name))
                score = sum(r["win_index"] for r in per_field) / len(per_field)
                ev = Evaluation(spec, score, per_field)
                self._cache[(seed, spec)] = ev
                if record:
                    self.history.append({
                        "label": label, "seed": seed, "spec": spec, "objective": score,
                        **{f"win_index_{'+'.join(f)}": r["win_index"]
                           for f, r in zip(self.fields, per_field, strict=True)},
                        **{f"round_score_{'+'.join(f)}": r["avg_round_score"]
                           for f, r in zip(self.fields, per_field, strict=True)},
                    })
        return [self._cache[(seed, s)] for s in specs]

    def run(
        self,
        params: list[str] | None = None,
        grid: dict[str, list[str]] | None = None,
        max_passes: int = 3,
        min_gain: float = 0.03,
        validation_games: int | None = None,
    ) -> OptimizeResult:
        grid = {**DEFAULT_GRID, **(grid or {})}
        params = params or relevant_params(self._profile(self.start))
        current = dict(self.start)
        cur = self.evaluate([current], self.seed, "départ")[0]
        self.log(f"Départ : {cur.spec}  objectif {cur.score:.3f}")
        result = OptimizeResult(build_spec(self.name, self.start), cur.spec)

        for pass_no in range(1, max_passes + 1):
            changed = False
            for param in params:
                if param not in grid:
                    raise ValueError(f"Pas de valeurs pour '{param}' : ajoutez --grid {param}=v1,v2,...")
                current_value = getattr(self._profile(current), param)
                candidates = [{**current, param: v} for v in grid[param] if _parse_value(v) != current_value]
                if not candidates:
                    continue
                evals = self.evaluate(candidates, self.seed, f"passe {pass_no} · {param}")
                best_i = max(range(len(evals)), key=lambda i: evals[i].score)
                if evals[best_i].score >= cur.score + min_gain:
                    msg = (f"passe {pass_no} : {param} {current_value} → {candidates[best_i][param]} "
                           f"(objectif {cur.score:.3f} → {evals[best_i].score:.3f})")
                    current, cur, changed = candidates[best_i], evals[best_i], True
                    result.steps.append(msg)
                    self.log(msg)
                else:
                    self.log(f"passe {pass_no} : {param} inchangé ({current_value}) — meilleure alternative "
                             f"{candidates[best_i][param]} à {evals[best_i].score:.3f}")
            if not changed:
                self.log(f"Passe {pass_no} sans amélioration : arrêt.")
                break

        result.best_spec = cur.spec
        # Validation sur une graine jamais vue : l'amélioration doit survivre hors de l'échantillon d'optimisation.
        v_seed = self.seed + 1_000_003
        configs = [dict(self.start)] + ([current] if current != self.start else [])
        result.validation = self.evaluate(configs, v_seed, "validation", validation_games or 2 * self.n_games,
                                          record=False)
        result.history = self.history
        return result
