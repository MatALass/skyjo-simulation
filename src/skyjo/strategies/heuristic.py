"""Moteur de décision paramétrable : chaque "personnalité" est un jeu de paramètres (Profile).

Principe : chaque coup possible reçoit une valeur = baisse attendue du score de manche
+ bonus/malus propres à la personnalité. Le joueur choisit le coup de plus grande valeur.

- Carte cachée = espérance des cartes inconnues (comptage) ou 5.07 (joueur qui ne compte pas).
- Compléter une colonne : on gagne aussi les deux autres cartes de la colonne.
- Une carte cachée vaut un peu moins que son espérance (`hidden_discount`) : si elle sort haute,
  on pourra encore la remplacer (valeur d'option).
- Finir la manche alors qu'on n'est pas devant coûte, en espérance, son propre score (doublement).
  Politique "probabilistic" : P(être strictement le plus bas) estimée par approximation normale
  du score final de chaque adversaire (somme de ses cartes cachées, variance issue du comptage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any

from skyjo.board import ROWS, Pos
from skyjo.cards import NAIVE_BELIEF, CardBelief
from skyjo.strategies.base import DrawDecision, GameView, Strategy

FINISH_POLICIES = ("always", "if_lowest", "probabilistic")
_TIE_EPS = 1e-6  # départage aléatoire des coups de même valeur


@dataclass(frozen=True)
class Profile:
    name: str
    description: str = ""
    # --- information ---
    card_counting: bool = False      # espérance des cartes cachées calculée sur les cartes encore inconnues
    draw_lookahead: bool = False     # compare la défausse à l'espérance (exacte) d'une pioche
    # --- placement ---
    hidden_max: int | None = 4       # carte max posée sur une carte cachée (None = dès que c'est rentable)
    hidden_discount: float = 0.0     # valeur d'option d'une carte cachée (retirée de son espérance)
    discard_min_gain: float = 3.0    # gain mini pour prendre la défausse (si pas de lookahead)
    pair_bonus: float = 0.0          # valeur d'une paire visible dans une colonne (espoir de colonne)
    # --- rythme ---
    reveal_bonus: float = 0.0        # valeur accordée au fait de retourner une carte (avancer vers la fin)
    tempo_weight: float = 0.0        # accélère si on mène, ralentit si on est derrière (lecture adverse)
    # --- adversaires ---
    deny_weight: float = 0.0         # pénalise la défausse d'une carte utile au joueur suivant
    # --- fin de manche ---
    finish_policy: str = "always"    # "always" | "if_lowest" | "probabilistic"
    finish_margin: float = 0.0       # if_lowest : avance exigée sur le meilleur adversaire estimé
    opp_last_turn_gain: float = 3.0  # probabilistic : amélioration attendue d'un adversaire à son dernier tour
    finish_bonus: float = 0.0        # envie de finir quand on est devant
    # --- début de manche ---
    initial_same_column: bool = False

    def __post_init__(self) -> None:
        if self.finish_policy not in FINISH_POLICIES:
            raise ValueError(f"finish_policy doit être dans {FINISH_POLICIES}")

    def with_overrides(self, overrides: dict[str, str]) -> Profile:
        valid = {f.name for f in fields(self)} - {"name", "description"}
        parsed: dict[str, Any] = {}
        for key, raw in overrides.items():
            if key not in valid:
                raise ValueError(f"Paramètre inconnu '{key}'. Possibles : {', '.join(sorted(valid))}")
            parsed[key] = _parse_value(raw)
        suffix = ";".join(f"{k}={v}" for k, v in overrides.items())
        return replace(self, name=f"{self.name}[{suffix}]", **parsed)


def _parse_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


@dataclass
class _Ctx:
    """Contexte d'une décision, calculé une fois."""

    belief: CardBelief
    hidden_value: float  # espérance d'une carte cachée
    hidden_place: float  # valeur d'une carte cachée quand on envisage de la remplacer
    my_est: float        # score final estimé (visibles + cachées à l'espérance)
    opp_best: float      # meilleur score final estimé parmi les adversaires
    tempo: float         # valeur effective d'un retournement
    # Instantané de mon plateau (évite de relire la vue des centaines de fois par décision).
    positions: list[Pos] = field(default_factory=list)
    hidden: list[Pos] = field(default_factory=list)
    cell: dict[Pos, int | None] = field(default_factory=dict)
    others: dict[Pos, list[int | None]] = field(default_factory=dict)
    deny_cache: dict[int, float] = field(default_factory=dict)
    next_board: list[tuple[int | None, list[int | None]]] | None = None


class HeuristicStrategy(Strategy):
    def __init__(self, profile: Profile) -> None:
        self.p = profile
        self.name = profile.name
        self.description = profile.description

    # ------------------------------------------------------------------ contexte
    def _ctx(self, view: GameView) -> _Ctx:
        belief = view.belief if self.p.card_counting else NAIVE_BELIEF
        e = belief.expected()
        b = view.board
        positions = b.positions()
        cell = {pos: b.peek(pos) for pos in positions}
        hidden = [pos for pos in positions if cell[pos] is None]
        others = {pos: b.column_others(pos) for pos in positions}
        my_est = sum(v for v in cell.values() if v is not None) + len(hidden) * e
        opp_best = min(b.visible_score() + b.hidden_count() * e for b in view.opponent_boards)
        if view.is_last_turn:
            tempo = 0.0  # plus rien à gagner à avancer : seul le score compte
        else:
            lead = max(-1.0, min(1.0, (opp_best - my_est) / 10))
            tempo = self.p.reveal_bonus + self.p.tempo_weight * lead
        hidden_place = e if view.is_last_turn else e - self.p.hidden_discount
        return _Ctx(belief, e, hidden_place, my_est, opp_best, tempo, positions, hidden, cell, others)

    # ------------------------------------------------------------------ évaluations
    def _p_strictly_lowest(self, view: GameView, ctx: _Ctx, my_final: float) -> float:
        """P(mon score < celui de chaque adversaire), adversaires supposés indépendants et ~normaux."""
        var = ctx.belief.variance()
        p = 1.0
        for b in view.opponent_boards:
            h = b.hidden_count()
            mu = b.visible_score() + h * ctx.hidden_value - self.p.opp_last_turn_gain
            sd = math.sqrt(h * var + 4.0)  # +4 : incertitude sur son dernier coup
            z = (my_final + 0.5 - mu) / sd  # P(adv >= mon score + 1)
            p *= 0.5 * math.erfc(z / math.sqrt(2))
        return p

    def _finish_value(self, view: GameView, ctx: _Ctx, my_final: float) -> float:
        policy = self.p.finish_policy
        if policy == "always":
            return 0.0
        if policy == "if_lowest":
            if my_final > 0 and my_final >= ctx.opp_best - self.p.finish_margin:
                return -my_final  # score doublé : coût = son propre score
            return self.p.finish_bonus
        p = self._p_strictly_lowest(view, ctx, my_final)
        doubling_cost = (1 - p) * my_final if my_final > 0 else 0.0
        return p * self.p.finish_bonus - doubling_cost

    def _finish_detail(self, view: GameView, ctx: _Ctx, my_final: float) -> dict[str, Any]:
        """Éléments chiffrés de la décision de finir (explication uniquement)."""
        d: dict[str, Any] = {"policy": self.p.finish_policy, "my_final": my_final, "opp_best": ctx.opp_best,
                             "margin": self.p.finish_margin, "bonus": self.p.finish_bonus}
        if self.p.finish_policy == "probabilistic":
            d["p_lowest"] = self._p_strictly_lowest(view, ctx, my_final)
        return d

    def _deny_cost(self, view: GameView, ctx: _Ctx, card: int) -> float:
        """Coût de laisser `card` sur la défausse pour le joueur suivant."""
        if not self.p.deny_weight:
            return 0.0
        cost = ctx.deny_cache.get(card)
        if cost is None:
            if ctx.next_board is None:  # instantané du plateau du joueur suivant, pris une fois
                nb = view.opponent_boards[0]
                ctx.next_board = [(nb.peek(pos), nb.column_others(pos)) for pos in nb.positions()]
            best = 0.0
            for current, others in ctx.next_board:
                cur = ctx.hidden_value if current is None else current
                gain = cur + 2 * card if (others[0] == card and others[1] == card) else cur - card
                if gain > best:
                    best = gain
            cost = ctx.deny_cache[card] = self.p.deny_weight * best
        return cost

    def _place_value(self, view: GameView, ctx: _Ctx, pos: Pos, card: int, enforce_limit: bool,
                     parts: dict[str, Any] | None = None) -> float | None:
        """Valeur de poser `card` en `pos` (l'ancienne carte part à la défausse).

        `parts` (explication) : rempli avec la décomposition de la valeur.
        """
        current = ctx.cell[pos]
        hidden = current is None
        if hidden and enforce_limit and self.p.hidden_max is not None and card > self.p.hidden_max:
            if parts is not None:
                parts["blocked"] = True
            return None
        cur = ctx.hidden_place if hidden else current
        others = ctx.others[pos]
        completes = others[0] == card and others[1] == card
        delta = cur + 2 * card if completes else cur - card  # baisse attendue du score
        value = delta
        if parts is not None:
            parts.update(current=current, hidden=hidden, cur_value=cur, completes=completes, gain=delta)
        if self.p.pair_bonus and not completes:
            if card in others:
                value += self.p.pair_bonus
                if parts is not None:
                    parts["pair"] = self.p.pair_bonus
            if not hidden and current in others:
                value -= self.p.pair_bonus  # on casse une paire existante
                if parts is not None:
                    parts["break_pair"] = -self.p.pair_bonus
        if hidden:
            value += ctx.tempo
            if parts is not None and ctx.tempo:
                parts["tempo"] = ctx.tempo
            if len(ctx.hidden) == 1 and not view.is_last_turn:
                fv = self._finish_value(view, ctx, ctx.my_est - delta)
                value += fv
                if parts is not None:
                    parts["finish"] = fv
                    parts["finish_detail"] = self._finish_detail(view, ctx, ctx.my_est - delta)
        else:
            dc = self._deny_cost(view, ctx, current)
            value -= dc
            if parts is not None and dc:
                parts["deny"] = -dc
        if parts is not None:
            parts["total"] = value
        return value

    def _reveal_value(self, view: GameView, ctx: _Ctx, pos: Pos, drawn: int,
                      parts: dict[str, Any] | None = None) -> float:
        """Valeur de défausser `drawn` et retourner `pos`."""
        dc = self._deny_cost(view, ctx, drawn)
        value = ctx.tempo - dc
        if parts is not None:
            if ctx.tempo:
                parts["tempo"] = ctx.tempo
            if dc:
                parts["deny"] = -dc
        others = ctx.others[pos]
        if others[0] is not None and others[0] == others[1]:
            x = others[0]
            chance = ctx.belief.prob(x) * 3 * x  # chance de compléter la colonne
            value += chance
            if parts is not None:
                parts["column_chance"] = chance
                parts["column_value"] = x
        if len(ctx.hidden) == 1 and not view.is_last_turn:
            fv = self._finish_value(view, ctx, ctx.my_est)
            value += fv
            if parts is not None:
                parts["finish"] = fv
                parts["finish_detail"] = self._finish_detail(view, ctx, ctx.my_est)
        if parts is not None:
            parts["total"] = value
        return value

    def _best_place(self, view: GameView, ctx: _Ctx, card: int, enforce_limit: bool = True) -> tuple[Pos | None, float]:
        best_pos, best_val = None, float("-inf")
        for pos in ctx.positions:
            v = self._place_value(view, ctx, pos, card, enforce_limit)
            if v is None:
                continue
            v += view.rng.random() * _TIE_EPS
            if v > best_val:
                best_pos, best_val = pos, v
        return best_pos, best_val

    def _best_reveal(self, view: GameView, ctx: _Ctx, drawn: int) -> tuple[Pos | None, float]:
        best_pos, best_val = None, float("-inf")
        for pos in ctx.hidden:
            v = self._reveal_value(view, ctx, pos, drawn) + view.rng.random() * _TIE_EPS
            if v > best_val:
                best_pos, best_val = pos, v
        return best_pos, best_val

    def _draw_ev(self, view: GameView, ctx: _Ctx) -> float:
        """Espérance de la valeur du meilleur coup après une pioche."""
        ev = 0.0
        for card, prob in ctx.belief.items():
            _, vp = self._best_place(view, ctx, card)
            _, vr = self._best_reveal(view, ctx, card)
            ev += prob * max(vp, vr)
        return ev

    # ------------------------------------------------------------------ explications (mode replay)
    def _ctx_summary(self, view: GameView, ctx: _Ctx) -> dict[str, Any]:
        return {"hidden_value": ctx.hidden_value, "hidden_place": ctx.hidden_place, "counting": self.p.card_counting,
                "unknown": ctx.belief.total, "my_est": ctx.my_est, "opp_best": ctx.opp_best, "tempo": ctx.tempo,
                "last_turn": view.is_last_turn}

    def _explain_place(self, view: GameView, ctx: _Ctx, pos: Pos, card: int) -> dict[str, Any]:
        parts: dict[str, Any] = {"kind": "place", "pos": pos}
        self._place_value(view, ctx, pos, card, enforce_limit=False, parts=parts)
        return parts

    def _explain_reveal(self, view: GameView, ctx: _Ctx, pos: Pos, card: int) -> dict[str, Any]:
        parts: dict[str, Any] = {"kind": "reveal", "pos": pos}
        self._reveal_value(view, ctx, pos, card, parts=parts)
        return parts

    def _options(self, view: GameView, ctx: _Ctx, card: int, with_reveal: bool) -> list[dict[str, Any]]:
        """Toutes les options chiffrées (sans tirage aléatoire), triées de la meilleure à la moins bonne."""
        opts: list[dict[str, Any]] = []
        for pos in ctx.positions:
            v = self._place_value(view, ctx, pos, card, enforce_limit=True)
            if v is not None:
                opts.append({"kind": "place", "pos": pos, "current": ctx.cell[pos], "value": v})
        if with_reveal:
            for pos in ctx.hidden:
                opts.append({"kind": "reveal", "pos": pos, "current": None,
                             "value": self._reveal_value(view, ctx, pos, card)})
        return sorted(opts, key=lambda o: -o["value"])

    def _params(self) -> dict[str, Any]:
        return {"hidden_max": self.p.hidden_max, "discard_min_gain": self.p.discard_min_gain,
                "lookahead": self.p.draw_lookahead, "finish_policy": self.p.finish_policy,
                "finish_margin": self.p.finish_margin}

    # ------------------------------------------------------------------ API
    def initial_reveals(self, view: GameView) -> list[Pos]:
        cols = view.board.columns_left()
        if self.explain:
            self.last_explanation = {"step": "initial", "same_column": self.p.initial_same_column}
        if self.p.initial_same_column:
            col = view.rng.choice(cols)
            return [(col, r) for r in view.rng.sample(range(ROWS), 2)]
        return [(c, view.rng.randrange(ROWS)) for c in view.rng.sample(cols, 2)]

    def take_discard(self, view: GameView) -> bool:
        if view.top_discard is None:
            return False
        ctx = self._ctx(view)
        card = view.top_discard
        pos, value = self._best_place(view, ctx, card)
        draw_ev = None
        if pos is None:
            take = False
        elif self.p.draw_lookahead:
            draw_ev = self._draw_ev(view, ctx)
            take = value > draw_ev
        else:
            take = value >= self.p.discard_min_gain
        if self.explain:
            self.last_explanation = {
                "step": "discard_choice", "card": card, "take": take, "blocked": pos is None,
                "best": self._explain_place(view, ctx, pos, card) if pos is not None else None,
                "draw_ev": draw_ev, "ctx": self._ctx_summary(view, ctx), "params": self._params(),
            }
        return take

    def place_discard(self, view: GameView, card: int) -> Pos:
        ctx = self._ctx(view)
        pos, _ = self._best_place(view, ctx, card)
        forced = pos is None
        if pos is None:
            pos, _ = self._best_place(view, ctx, card, enforce_limit=False)
        assert pos is not None
        if self.explain:
            self.last_explanation = {
                "step": "place_discard", "card": card, "forced": forced,
                "chosen": self._explain_place(view, ctx, pos, card),
                "options": self._options(view, ctx, card, with_reveal=False),
                "ctx": self._ctx_summary(view, ctx), "params": self._params(),
            }
        return pos

    def handle_drawn(self, view: GameView, card: int) -> DrawDecision:
        ctx = self._ctx(view)
        pos_p, vp = self._best_place(view, ctx, card)
        pos_r, vr = self._best_reveal(view, ctx, card)
        if pos_r is None or (pos_p is not None and vp >= vr):
            forced = pos_p is None
            if pos_p is None:
                pos_p, _ = self._best_place(view, ctx, card, enforce_limit=False)
            assert pos_p is not None
            decision = DrawDecision(True, pos_p)
        else:
            forced = False
            decision = DrawDecision(False, pos_r)
        if self.explain:
            self.last_explanation = {
                "step": "drawn", "card": card, "replace": decision.replace, "forced": forced,
                "place": self._explain_place(view, ctx, pos_p, card) if pos_p is not None else None,
                "reveal": self._explain_reveal(view, ctx, pos_r, card) if pos_r is not None else None,
                "options": self._options(view, ctx, card, with_reveal=True),
                "ctx": self._ctx_summary(view, ctx), "params": self._params(),
            }
        return decision
