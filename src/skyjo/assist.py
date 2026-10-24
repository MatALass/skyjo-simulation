"""Assistant en direct : conseil de l'Expert pour une situation saisie à la main.

- `advise()` : la décision et son explication chiffrée (moteur Python de référence).
- `parity_cases()` : situations aléatoires + réponses Python, pour vérifier que le portage JavaScript
  (web/engine.js, utilisé par la page assistant.html) calcule exactement la même chose.
- `build_assistant()` : produit la page HTML autonome de l'assistant (moteur JS + profils exportés de Python).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Any

from skyjo.board import COLS, ROWS
from skyjo.cards import build_deck
from skyjo.coach import Grid, Situation, _deal
from skyjo.game import Round
from skyjo.strategies import PROFILES, make_strategy


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def advise(me: Grid, opponents: list[Grid], discard: int | None, drawn: int | None = None,
           profile: str = "expert") -> dict[str, Any]:
    """Conseil pour mon tour. Sans `drawn` : prendre la défausse ou piocher. Avec : quoi faire de la carte."""
    situation = Situation(me, tuple(opponents), discard, drawn)
    strat = make_strategy(profile)
    strat.explain = True
    others = [make_strategy("greedy") for _ in opponents]
    boards, pile = _deal(situation, random.Random(0))
    rnd = Round.from_position([strat, *others], random.Random(0), boards, pile,
                              [] if discard is None else [discard])
    last = situation.finisher is not None
    if drawn is None:
        view = rnd._view(0, last, 0)
        take = strat.take_discard(view)
        exp = dict(strat.last_explanation or {})
        ctx = strat._ctx(view)
        exp["options"] = strat._options(view, ctx, discard, with_reveal=False)
        if take:
            card = rnd.piles.take_discard()
            strat.place_discard(rnd._view(0, last, 0, in_hand=card), card)
            exp["placement"] = strat.last_explanation
        return _jsonable(exp)
    strat.handle_drawn(rnd._view(0, last, 0, in_hand=drawn), drawn)
    return _jsonable(strat.last_explanation)


# ---------------------------------------------------------------------------------------------- parité JS
def _random_grid(deck: list[int], rng: random.Random, hidden_min: int, hidden_max: int,
                 remove_prob: float) -> Grid:
    cols: list[list[int] | None] = []
    for _ in range(COLS):
        if rng.random() < remove_prob:
            cols.append(None)  # colonne retirée : ses cartes sont sorties du jeu (défausse profonde)
            for _ in range(ROWS):
                deck.pop()
        else:
            cols.append([deck.pop() for _ in range(ROWS)])
    cells = [(c, r) for c, col in enumerate(cols) if col is not None for r in range(ROWS)]
    if not cells:
        return _random_grid(deck, rng, hidden_min, hidden_max, 0.0)
    n_hidden = rng.randint(min(hidden_min, len(cells)), min(hidden_max, len(cells)))
    hidden = set(rng.sample(cells, n_hidden))
    return [None if col is None else [None if (c, r) in hidden else col[r] for r in range(ROWS)]
            for c, col in enumerate(cols)]


def parity_cases(n: int = 300, seed: int = 0, profiles: list[str] | None = None) -> list[dict[str, Any]]:
    """Situations aléatoires variées (paires, dernière carte cachée, dernier tour, colonnes retirées)."""
    rng = random.Random(seed)
    profiles = profiles or list(PROFILES)
    cases = []
    while len(cases) < n:
        deck = build_deck()
        rng.shuffle(deck)
        n_opp = rng.randint(1, 4)
        me_hidden = 1 if rng.random() < 0.25 else rng.randint(1, 10)
        me = _random_grid(deck, rng, me_hidden, me_hidden, 0.08)
        last_turn = rng.random() < 0.2
        opponents = []
        for i in range(n_opp):
            lo, hi = (0, 0) if (last_turn and i == 0) else (1, 10)
            opponents.append(_random_grid(deck, rng, lo, hi, 0.08))
        if rng.random() < 0.3:  # forcer une paire visible dans ma grille
            for col in me:
                if col and col[0] is not None and col[1] is None:
                    col[1] = col[0]
                    break
        discard = deck.pop()
        drawn = deck.pop() if rng.random() < 0.5 else None
        try:
            Situation(me, tuple(opponents), discard, drawn)
        except ValueError:
            continue  # paire forcée impossible avec les cartes restantes
        expected = {p: advise(me, opponents, discard, drawn, profile=p) for p in profiles}
        cases.append({"me": me, "opponents": opponents, "discard": discard, "drawn": drawn, "expected": expected})
    return cases


# ---------------------------------------------------------------------------------------------- page
def profiles_json() -> dict[str, dict[str, Any]]:
    return {name: asdict(p) for name, p in PROFILES.items()}


def _web(name: str) -> str:
    return resources.files("skyjo.web").joinpath(name).read_text(encoding="utf-8")


def build_assistant(standalone: bool = True) -> str:
    engine = _web("engine.js")
    page = _web("assistant.html")
    page = page.replace("/*__ENGINE__*/", engine.replace("</", "<\\/"))
    page = page.replace("/*__PROFILES__*/null", json.dumps(profiles_json(), ensure_ascii=False))
    if not standalone:
        return page
    return ('<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + page + "\n</body>\n</html>\n")


def write_assistant(out: str | Path, standalone: bool = True) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_assistant(standalone), encoding="utf-8")
    return out
