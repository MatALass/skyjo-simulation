import random

import pytest

from skyjo.board import Board, Slot
from skyjo.coach import Situation, coach, legal_moves, parse_grid
from skyjo.game import Round
from skyjo.optimize import Optimizer, build_spec, fields_for_players, relevant_params, split_spec
from skyjo.strategies import make_strategy

ME = "5 ? ? 3 / ? 12 ? ? / 0 ? ? ?"
OPP = "? 4 ? ? / ? ? ? 1 / ? ? 2 ?"


# ---------------------------------------------------------------------------------------------- saisie
def test_parse_grid_with_removed_column():
    grid = parse_grid("5 ? x 3 / ? 12 x ? / 0 ? x ?")
    assert grid[2] is None
    assert grid[0] == [5, None, 0]
    assert grid[1] == [None, 12, None]


BAD_GRIDS = ["5 ? ? / ? ? ? ? / ? ? ? ?", "5 x ? ? / ? ? ? ? / ? ? ? ?", "13 ? ? ? / ? ? ? ? / ? ? ? ?"]


@pytest.mark.parametrize("bad", BAD_GRIDS)
def test_parse_grid_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_grid(bad)


def test_situation_rejects_impossible_card_counts():
    with pytest.raises(ValueError):  # il n'existe que 5 cartes -2
        Situation(parse_grid("-2 -2 -2 ? / -2 -2 -2 ? / ? ? ? ?"), (parse_grid(OPP),), top_discard=3)


def test_legal_moves():
    s = Situation(parse_grid(ME), (parse_grid(OPP),), top_discard=7)
    assert len(legal_moves(s)) == 12 + 1  # prendre la défausse sur chaque case, ou piocher
    s = Situation(parse_grid(ME), (parse_grid(OPP),), top_discard=7, drawn=9)
    assert len(legal_moves(s)) == 12 + 8  # poser sur chaque case, ou retourner une des 8 cachées


def test_finisher_detected_from_fully_visible_opponent():
    s = Situation(parse_grid(ME), (parse_grid(OPP), parse_grid(FINISHED_OPP)), top_discard=7)
    assert s.finisher == 2


# ---------------------------------------------------------------------------------------------- moteur
def test_round_resumed_with_no_turn_left_is_scored_immediately():
    boards = [Board.from_columns([[Slot(1, True)] * 3] * 4), Board.from_columns([[Slot(2, True)] * 3] * 4)]
    strategies = [make_strategy("greedy"), make_strategy("greedy")]
    rnd = Round.from_position(strategies, random.Random(0), boards, [5] * 20, [3])
    res = rnd.continue_from(0, finisher=0, remaining=0)
    assert res.turns == 0
    assert res.raw_scores == [0, 0]  # colonnes identiques retirées au retournement final


# ---------------------------------------------------------------------------------------------- coach
FINISHED_OPP = "1 2 3 4 / 0 1 2 3 / 4 0 1 2"


def test_coach_finds_obvious_move_on_last_turn():
    # Dernier tour (l'adversaire a fini) : plus de futur, le -2 doit remplacer le 12 (gain 14 > 7).
    s = Situation(parse_grid(ME), (parse_grid(FINISHED_OPP),), top_discard=7, drawn=-2)
    results = coach(s, samples=60, batch=30, policy="greedy")
    assert results[0].move.kind == "replace" and results[0].move.pos == (1, 1)
    assert results[0].n == 60
    assert all(r.diff_mean >= 0 for r in results[1:] if r.eliminated_at is None)


def test_coach_prefers_placing_a_minus_two_over_discarding_it():
    s = Situation(parse_grid(ME), (parse_grid(OPP),), top_discard=7, drawn=-2)
    results = coach(s, samples=60, batch=30, policy="greedy")
    assert results[0].move.kind == "replace"


def test_coach_last_turn_and_start_of_turn_run():
    s = Situation(parse_grid(ME), (parse_grid(OPP), parse_grid(FINISHED_OPP)), top_discard=2)
    results = coach(s, samples=20, batch=20, policy="greedy", prune=False)
    assert len(results) == 13
    assert all(r.finished == 0 for r in results)  # impossible de finir : un adversaire a déjà fini


def test_coach_is_reproducible():
    s = Situation(parse_grid(ME), (parse_grid(OPP),), top_discard=7, drawn=4)
    a = coach(s, samples=20, batch=20, policy="greedy", seed=3)
    b = coach(s, samples=20, batch=20, policy="greedy", seed=3)
    assert [(r.label, r.mean) for r in a] == [(r.label, r.mean) for r in b]


# ---------------------------------------------------------------------------------------------- optimize
def test_spec_round_trip():
    name, ov = split_spec("prudent:reveal_bonus=0:hidden_max=4")
    assert name == "prudent" and ov == {"reveal_bonus": "0", "hidden_max": "4"}
    assert build_spec(name, ov) == "prudent:hidden_max=4:reveal_bonus=0"


def test_relevant_params_depend_on_profile():
    assert "discard_min_gain" not in relevant_params(make_strategy("stratege").p)
    assert "finish_margin" in relevant_params(make_strategy("prudent").p)
    assert "finish_margin" not in relevant_params(make_strategy("fonceur").p)


def test_optimizer_small_run():
    logs = []
    opt = Optimizer("prudent", [["fonceur"]], n_games=20, seed=1, log=logs.append)
    res = opt.run(params=["hidden_max"], grid={"hidden_max": ["1", "4"]}, max_passes=1, validation_games=20)
    assert res.best_spec.startswith("prudent")
    assert 1 <= len(res.validation) <= 2
    assert {h["spec"] for h in res.history} >= {"prudent", "prudent:hidden_max=1", "prudent:hidden_max=4"}
    assert logs


def test_fields_for_players():
    assert len(fields_for_players([2])) == 4
    assert len(fields_for_players([3])) == 6
    assert len(fields_for_players([4])) == 4
    assert all(len(f) == 3 for f in fields_for_players([4]))
    with pytest.raises(ValueError):
        fields_for_players([6])


def test_expert_matches_optimized_spec():
    from dataclasses import replace

    spec = ("prudent:discard_min_gain=5:finish_margin=10:hidden_discount=1:hidden_max=none:pair_bonus=4:"
            "pair_scale=0.25:reveal_bonus=0.5:reveal_spread=4")
    optimized = make_strategy(spec).p
    expert = make_strategy("expert").p
    assert replace(optimized, name="", description="") == replace(expert, name="", description="")


# ---------------------------------------------------------------------------------------------- nouveaux paramètres
def _ctx_for(spec, me_grid, drawn=5):
    from skyjo.coach import Situation, _deal
    from skyjo.strategies.base import GameView

    s = Situation(parse_grid(me_grid), (parse_grid(OPP),), top_discard=7, drawn=drawn)
    boards, pile = _deal(s, random.Random(0))
    rnd = Round.from_position([make_strategy(spec), make_strategy("greedy")], random.Random(0), boards, pile, [7])
    view: GameView = rnd._view(0, False, 0, in_hand=drawn)
    strat = rnd.strategies[0]
    return strat, view, strat._ctx(view)


def test_pair_scale_values_high_pairs_more():
    grid = "9 1 ? ? / ? ? ? ? / ? ? ? ?"
    strat, view, ctx = _ctx_for("expert:pair_bonus=0:pair_scale=1", grid)
    high = strat._place_value(view, ctx, (0, 1), 9, enforce_limit=False, parts=(p9 := {}))
    low = strat._place_value(view, ctx, (1, 1), 1, enforce_limit=False, parts=(p1 := {}))
    assert p9["pair"] == 9 and p1["pair"] == 1
    assert high is not None and low is not None


def test_pair_min_blocks_pairs_of_negative_cards():
    grid = "-2 ? ? ? / ? ? ? ? / ? ? ? ?"
    strat, view, ctx = _ctx_for("expert:pair_min=1", grid)
    parts = {}
    strat._place_value(view, ctx, (0, 1), -2, enforce_limit=False, parts=parts)
    assert "pair" not in parts


def test_reveal_spread_prefers_columns_with_more_hidden_cards():
    grid = "5 ? ? ? / 6 ? ? ? / ? ? ? ?"  # colonne 1 : 1 cachée ; colonne 2 : 3 cachées
    strat, view, ctx = _ctx_for("expert:reveal_spread=2", grid)
    last_hidden = strat._reveal_value(view, ctx, (0, 2), 12)
    fresh = strat._reveal_value(view, ctx, (1, 0), 12)
    assert fresh - last_hidden == pytest.approx(2.0)
