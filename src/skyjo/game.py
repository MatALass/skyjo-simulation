"""Moteur de jeu : une manche, puis une partie complète."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from skyjo.board import Board, BoardView, Pos
from skyjo.cards import CARD_DISTRIBUTION, CardBelief, Piles
from skyjo.rules import DEFAULT_RULES, Rules
from skyjo.strategies.base import GameView, Strategy

# Observateur (replay) : appelé avec (événement, manche, données). Événements : "deal", "initial", "start",
# "turn", "round_end" (et "game_end" depuis play_game). Sans observateur, aucun coût.
Observer = Callable[[str, "Round", dict[str, Any]], None]


@dataclass
class RoundResult:
    raw_scores: list[int]        # score avant pénalité
    scores: list[int]            # score après doublement éventuel du finisseur
    finisher: int | None         # None si la manche a été bloquée
    first_player: int
    turns: int
    stalled: bool = False

    @property
    def finisher_doubled(self) -> bool:
        return self.finisher is not None and self.scores[self.finisher] != self.raw_scores[self.finisher]


@dataclass
class GameResult:
    totals: list[int]
    winners: list[int]           # plusieurs sièges en cas d'égalité
    rounds: list[RoundResult] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)


class Round:
    def __init__(
        self,
        strategies: list[Strategy],
        rng: random.Random,
        rules: Rules = DEFAULT_RULES,
        first_player: int | None = None,
        record: bool = False,
        boards: list[list[int]] | None = None,
        observer: Observer | None = None,
    ) -> None:
        """`boards` : plateaux imposés (12 cartes par joueur) pour rejouer une situation précise."""
        if not 2 <= len(strategies) <= 8:
            raise ValueError("Skyjo se joue de 2 à 8 joueurs.")
        self.strategies = strategies
        self.rng = rng
        self.rules = rules
        self.forced_first = first_player
        self.piles = Piles(rng)
        if boards is None:
            boards = [[self.piles.draw() for _ in range(12)] for _ in strategies]
        else:
            if len(boards) != len(strategies):
                raise ValueError("Un plateau imposé par joueur.")
            for card in (c for b in boards for c in b):
                self.piles.draw_pile.remove(card)  # ValueError si le paquet ne contient pas la carte
        self.boards = [Board(b) for b in boards]
        self.views = [BoardView(b) for b in self.boards]
        self.piles.discard(self.piles.draw())
        self.decisions: list[dict[str, Any]] | None = [] if record else None
        self.observer = observer

    # ------------------------------------------------------------------ information
    def _belief(self, in_hand: int | None = None) -> CardBelief:
        counts = dict(CARD_DISTRIBUTION)
        for b in self.boards:
            for p in b.visible_positions():
                counts[b.slot(p).value] -= 1
        for c in self.piles.discard_pile:
            counts[c] -= 1
        if in_hand is not None:
            counts[in_hand] -= 1
        return CardBelief(counts)

    def _view(self, idx: int, is_last_turn: bool, turn: int, in_hand: int | None = None) -> GameView:
        n = len(self.boards)
        return GameView(
            board=self.views[idx],
            opponent_boards=tuple(self.views[(idx + k) % n] for k in range(1, n)),
            top_discard=self.piles.top_discard,
            belief=self._belief(in_hand),
            is_last_turn=is_last_turn,
            turn=turn,
            rng=self.rng,
        )

    def _check_pos(self, idx: int, pos: Pos) -> None:
        if not self.boards[idx].has(pos):
            raise ValueError(f"{self.strategies[idx].name} : position illégale {pos}")

    # ------------------------------------------------------------------ déroulé
    def _setup(self) -> int:
        """Chaque joueur retourne 2 cartes. Manche 1 : plus gros total commence (égalité tirée au sort)."""
        if self.observer:
            self.observer("deal", self, {})
        for i, strat in enumerate(self.strategies):
            strat.last_explanation = None
            positions = strat.initial_reveals(self._view(i, False, 0))
            if len(set(positions)) != 2:
                raise ValueError(f"{strat.name} doit retourner exactement 2 cartes distinctes.")
            for p in positions:
                self._check_pos(i, p)
                self.boards[i].reveal(p)
            if self.observer:
                self.observer("initial", self, {"player": i, "positions": list(positions),
                                                "explanation": strat.last_explanation})
        totals = [b.visible_score() for b in self.boards]
        if self.forced_first is not None:
            first = self.forced_first
        else:
            best = max(totals)
            first = self.rng.choice([i for i, t in enumerate(totals) if t == best])
        if self.observer:
            self.observer("start", self, {"first": first, "totals": totals, "forced": self.forced_first is not None})
        return first

    def _snapshot(self, idx: int) -> dict[str, Any]:
        board = self.boards[idx]
        e = self._belief().expected()
        opps = [b for i, b in enumerate(self.boards) if i != idx]
        my_est = board.visible_score() + board.hidden_count() * e
        opp_best = min(b.visible_score() + b.hidden_count() * e for b in opps)
        return {
            "seat": idx,
            "strategy": self.strategies[idx].name,
            "n_players": len(self.boards),
            "hidden_before": board.hidden_count(),
            "visible_score_before": board.visible_score(),
            "expected_hidden": round(e, 3),
            "my_est_before": round(my_est, 2),
            "opp_best_est": round(opp_best, 2),
            "lead_before": round(opp_best - my_est, 2),  # > 0 : je suis devant
            "opp_min_hidden": min(b.hidden_count() for b in opps),
            "top_discard_before": self.piles.top_discard,
        }

    def apply_action(self, idx: int, source: str, card: int, pos: Pos, replace: bool) -> dict[str, Any]:
        """Applique un coup déjà décidé (la carte a déjà quitté la pioche ou la défausse)."""
        board = self.boards[idx]
        self._check_pos(idx, pos)
        target_was_visible = board.slot(pos).visible
        if replace:
            target_value = board.replace(pos, card)
            self.piles.discard(target_value)
            action = "replace_visible" if target_was_visible else "replace_hidden"
        else:
            if target_was_visible:
                raise ValueError(f"{self.strategies[idx].name} : impossible de retourner une carte visible {pos}")
            self.piles.discard(card)
            target_value = board.reveal(pos)
            action = "reveal"

        removed: list[int] = []
        if replace or self.rules.columns_on_reveal:
            removed = board.remove_completed_columns(only=[pos[0]])
            for c in removed:  # le triplet va sur la défausse après la carte échangée
                self.piles.discard(c)
        return {
            "source": source,
            "card": card,
            "pos": pos,
            "removed": removed,
            "action": action,
            "target_was_visible": target_was_visible,
            "target_value": target_value,
            "column_removed": bool(removed),
        }

    def _play_turn(self, idx: int, is_last_turn: bool, turn: int) -> None:
        strat = self.strategies[idx]
        record = self._snapshot(idx) if self.decisions is not None else None
        explanations: list[dict[str, Any] | None] = []
        strat.last_explanation = None

        if self.piles.top_discard is not None and strat.take_discard(self._view(idx, is_last_turn, turn)):
            explanations.append(strat.last_explanation)
            strat.last_explanation = None
            card = self.piles.take_discard()
            pos = strat.place_discard(self._view(idx, is_last_turn, turn, in_hand=card), card)
            info = self.apply_action(idx, "discard", card, pos, replace=True)
        else:
            explanations.append(strat.last_explanation)
            strat.last_explanation = None
            card = self.piles.draw()
            decision = strat.handle_drawn(self._view(idx, is_last_turn, turn, in_hand=card), card)
            info = self.apply_action(idx, "draw", card, decision.pos, decision.replace)
        explanations.append(strat.last_explanation)

        if self.observer:
            self.observer("turn", self, {
                "player": idx, "turn": turn, "is_last_turn": is_last_turn, "info": info,
                "explanations": explanations,
                "finished_round": self.boards[idx].all_visible() and not is_last_turn,
            })

        if record is not None:
            assert self.decisions is not None
            record.update(
                turn=turn,
                is_last_turn=is_last_turn,
                finished_round=self.boards[idx].all_visible() and not is_last_turn,
                **{k: v for k, v in info.items() if k not in ("pos", "removed")},
            )
            self.decisions.append(record)

    def play(self) -> RoundResult:
        first = self._setup()
        return self.continue_from(first, first_player=first)

    def continue_from(
        self,
        idx: int,
        finisher: int | None = None,
        remaining: int = 0,
        turns: int = 0,
        first_player: int = 0,
    ) -> RoundResult:
        """Joue la manche à partir du joueur `idx`.

        `finisher` / `remaining` : si un joueur a déjà tout retourné, nombre de tours restant à jouer
        (0 = la manche est terminée, on passe directement au décompte).
        """
        n = len(self.strategies)
        stalled = False
        while not (finisher is not None and remaining <= 0):
            self._play_turn(idx, finisher is not None, turns)
            turns += 1
            if finisher is None:
                if self.boards[idx].all_visible():
                    finisher, remaining = idx, n - 1
            else:
                remaining -= 1
            if turns >= self.rules.max_turns_per_round:
                stalled = True
                break
            idx = (idx + 1) % n
        return self._score(finisher, first_player, turns, stalled)

    def _score(self, finisher: int | None, first_player: int, turns: int, stalled: bool) -> RoundResult:
        final_revealed: list[list[Pos]] = []
        final_removed: list[list[int]] = []
        for b in self.boards:
            final_revealed.append(b.hidden_positions())
            cols_before = b.column_indices()
            b.reveal_all()
            if self.rules.columns_at_final_reveal:
                b.remove_completed_columns()
            final_removed.append([c for c in cols_before if c not in b.column_indices()])

        raw = [b.total_score() for b in self.boards]
        scores = list(raw)
        if finisher is not None and not stalled:
            others_min = min(s for i, s in enumerate(raw) if i != finisher)
            # Doublé si pas strictement le plus bas (égalité incluse), seulement si score positif.
            if raw[finisher] > 0 and raw[finisher] >= others_min:
                scores[finisher] *= 2
        else:
            finisher = None

        result = RoundResult(raw, scores, finisher, first_player, turns, stalled)
        if self.observer:
            self.observer("round_end", self, {"result": result, "final_revealed": final_revealed,
                                              "final_removed": final_removed})
        if self.decisions is not None:
            best = min(scores)
            for d in self.decisions:
                s = d["seat"]
                d.update(
                    round_score=scores[s],
                    raw_round_score=raw[s],
                    won_round=scores[s] == best,
                    round_rank=1 + sum(x < scores[s] for x in scores),
                    is_finisher=s == finisher,
                    finisher_doubled=s == finisher and result.finisher_doubled,
                )
        return result

    @classmethod
    def from_position(
        cls,
        strategies: list[Strategy],
        rng: random.Random,
        boards: list[Board],
        draw_pile: list[int],
        discard_pile: list[int],
        rules: Rules = DEFAULT_RULES,
    ) -> Round:
        """Manche reprise en cours de route (plateaux déjà en place, pioche et défausse données)."""
        if len(boards) != len(strategies):
            raise ValueError("Un plateau par joueur.")
        rnd = cls.__new__(cls)
        rnd.strategies = strategies
        rnd.rng = rng
        rnd.rules = rules
        rnd.forced_first = None
        rnd.piles = Piles(rng)
        rnd.piles.draw_pile = list(draw_pile)
        rnd.piles.discard_pile = list(discard_pile)
        rnd.boards = boards
        rnd.views = [BoardView(b) for b in boards]
        rnd.decisions = None
        rnd.observer = None
        return rnd


def play_game(
    strategies: list[Strategy],
    rng: random.Random,
    rules: Rules = DEFAULT_RULES,
    record: bool = False,
    observer: Observer | None = None,
) -> GameResult:
    totals = [0] * len(strategies)
    rounds: list[RoundResult] = []
    decisions: list[dict[str, Any]] = []
    first: int | None = None  # manche 1 : règle des 2 cartes ; ensuite : le finisseur commence
    while max(totals) < rules.end_score:
        if len(rounds) >= rules.max_rounds_per_game:
            raise RuntimeError("Partie trop longue.")
        rnd = Round(strategies, rng, rules, first_player=first, record=record, observer=observer)
        result = rnd.play()
        if rnd.decisions:
            for d in rnd.decisions:
                d["round"] = len(rounds)
            decisions.extend(rnd.decisions)
        rounds.append(result)
        totals = [t + s for t, s in zip(totals, result.scores, strict=True)]
        first = result.finisher
    best = min(totals)
    result = GameResult(totals, [i for i, t in enumerate(totals) if t == best], rounds, decisions)
    if observer:
        observer("game_end", rnd, {"result": result})
    return result
