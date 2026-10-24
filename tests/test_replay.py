import json
import random

import pytest

from skyjo.game import play_game
from skyjo.replay import num, record_game, render_html
from skyjo.strategies import make_strategy

SPECS = ["expert", "prudent", "fonceur", "collectionneur"]


def test_explain_mode_does_not_change_the_game():
    """Le mode explication ne doit ni changer une décision ni consommer de tirage aléatoire."""
    plain = play_game([make_strategy(s) for s in SPECS], random.Random(11))
    replay = record_game(SPECS, seed=11)
    assert replay["meta"]["roundScores"] == [r.scores for r in plain.rounds]


@pytest.mark.parametrize("specs", [SPECS, ["stratege", "random"], ["opportuniste", "greedy", "expert"]])
def test_replay_structure(specs):
    replay = record_game(specs, seed=3)
    steps = replay["steps"]
    kinds = [s["kind"] for s in steps]
    assert kinds[0] == "deal" and kinds[-1] == "game_end"
    assert kinds.count("deal") == kinds.count("round_end") == replay["meta"]["rounds"]
    assert kinds.count("initial") == len(specs) * replay["meta"]["rounds"]
    for s in steps:
        assert s["title"] and isinstance(s["details"], list)
        assert len(s["state"]["boards"]) == len(specs)
    turns = [s for s in steps if s["kind"] == "turn"]
    assert all(s["details"] for s in turns)  # chaque coup est justifié
    assert sum(s["state"]["unknown"][str(v)] for v in range(-2, 13) for s in steps[:1]) == 150 - 1


def test_first_turn_state_is_consistent():
    steps = record_game(SPECS, seed=5)["steps"]
    initial = [s for s in steps if s["kind"] == "initial"][-1]
    visible = sum(vis for board in initial["state"]["boards"] for col in board for _, vis in col)
    assert visible == 2 * len(SPECS)


def test_render_html_embeds_data():
    replay = record_game(["expert", "prudent"], seed=1)
    page = render_html(replay)
    assert page.startswith("<!doctype html>") and "<title>Skyjo pas à pas</title>" in page
    assert "/*__REPLAY_DATA__*/" not in page
    start = page.index("const REPLAY = ") + len("const REPLAY = ")
    end = page.index(";\n", start)
    assert json.loads(page[start:end].replace("<\\/", "</"))["meta"]["players"][0]["label"] == "Expert"
    fragment = render_html(replay, standalone=False)
    assert not fragment.startswith("<!doctype")


def test_num_formats_french():
    assert num(4.0) == "4" and num(4.25) == "4,2" or num(4.25) == "4,3"
    assert num(-3.5) == "−3,5" and num(0) == "0"
