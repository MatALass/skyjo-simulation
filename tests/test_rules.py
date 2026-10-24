import random
from itertools import pairwise

import pytest

from skyjo.board import Board, BoardView
from skyjo.cards import DECK_SIZE, CardBelief, Piles, build_deck
from skyjo.game import Round, play_game
from skyjo.rules import Rules
from skyjo.strategies import RandomStrategy, make_strategy
from skyjo.strategies.base import DrawDecision, Strategy


def test_deck_composition():
    deck = build_deck()
    assert len(deck) == DECK_SIZE == 150
    assert deck.count(-2) == 5 and deck.count(-1) == 10 and deck.count(0) == 15
    assert all(deck.count(v) == 10 for v in range(1, 13))


def test_column_removed_when_three_identical_visible():
    board = Board([7, 7, 7] + [1] * 9)
    assert board.remove_completed_columns() == []
    for r in range(3):
        board.reveal((0, r))
    assert board.remove_completed_columns(only=[1, 2, 3]) == []
    assert board.remove_completed_columns(only=[0]) == [7, 7, 7]
    assert len(board.positions()) == 9 and board.total_score() == 9


def test_board_view_hides_cards():
    board = Board(list(range(12)))
    view = BoardView(board)
    with pytest.raises(PermissionError):
        view.value((0, 0))
    assert view.peek((0, 0)) is None
    board.reveal((0, 0))
    assert view.value((0, 0)) == 0
    assert view.column_others((0, 1)) == [0, None]


def test_belief_counts_known_cards():
    belief = CardBelief({-2: 5, 12: 10})
    assert belief.total == 15
    assert belief.prob(-2) == pytest.approx(1 / 3)
    assert belief.expected() == pytest.approx((5 * -2 + 120) / 15)


def test_round_belief_excludes_visible_and_discard():
    rnd = Round([RandomStrategy(), RandomStrategy()], random.Random(0))
    rnd._setup()
    belief = rnd._belief()
    assert belief.total == 150 - 4 - 1  # 2 cartes retournées par joueur + défausse


class _Scripted(Strategy):
    """Pioche toujours et retourne la première carte cachée (scénarios déterministes)."""

    name = "scripted"

    def initial_reveals(self, view):
        return view.board.hidden_positions()[:2]

    def take_discard(self, view):
        return False

    def place_discard(self, view, card):
        raise AssertionError

    def handle_drawn(self, view, card):
        return DrawDecision(False, view.board.hidden_positions()[0])


DEFAULT = Rules()


def _rigged_round(cards_by_player, first=0, rules=DEFAULT):
    # Le joueur 0 commence et retourne une carte par tour : il finit toujours le premier.
    return Round([_Scripted() for _ in cards_by_player], random.Random(0), rules=rules,
                 first_player=first, boards=cards_by_player)


def test_finisher_doubled_when_not_strictly_lowest():
    res = _rigged_round([[1, 2, 3] * 4, [0, 0, 1] * 4]).play()
    assert res.finisher == 0
    assert res.raw_scores == [24, 4] and res.scores == [48, 4]


def test_finisher_never_doubled_when_negative():
    p1 = [-2, -2, -1, -2, -2, -1, -2, -1, -1, 0, -1, -1]  # -16
    res = _rigged_round([[-1, 0, 0] * 4, p1]).play()
    assert res.finisher == 0
    assert res.raw_scores == [-4, -16] and res.scores[0] == -4


P1_COLUMNS = [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]


def test_columns_removed_during_play_and_at_final_reveal():
    assert _rigged_round([[1, 2, 3] * 4, P1_COLUMNS]).play().raw_scores == [24, 0]
    # Sans retrait pendant le jeu, le retournement final retire quand même les colonnes.
    only_final = Rules(columns_on_reveal=False)
    assert _rigged_round([[1, 2, 3] * 4, P1_COLUMNS], rules=only_final).play().raw_scores == [24, 0]
    # Variante sans aucun retrait par retournement.
    none = Rules(columns_on_reveal=False, columns_at_final_reveal=False)
    assert _rigged_round([[1, 2, 3] * 4, P1_COLUMNS], rules=none).play().raw_scores == [24, 78]


def test_next_round_starts_with_previous_finisher():
    game = play_game([make_strategy("fonceur"), make_strategy("prudent")], random.Random(3))
    for prev, nxt in pairwise(game.rounds):
        if prev.finisher is not None:
            assert nxt.first_player == prev.finisher


def test_reshuffle_keeps_top_discard():
    piles = Piles(random.Random(0))
    piles.discard_pile = piles.draw_pile[:]
    piles.draw_pile = []
    top = piles.top_discard
    piles.draw()
    assert piles.discard_pile == [top]


ALL_SPECS = ["prudent", "fonceur", "collectionneur", "opportuniste", "stratege", "greedy", "random"]


@pytest.mark.parametrize("spec", ALL_SPECS)
@pytest.mark.parametrize("n_players", [2, 4, 8])
def test_every_strategy_plays_full_games(spec, n_players):
    strategies = [make_strategy(spec)] + [make_strategy("greedy") for _ in range(n_players - 1)]
    game = play_game(strategies, random.Random(n_players), record=True)
    assert max(game.totals) >= 100
    assert game.decisions and all("round_score" in d for d in game.decisions)


def test_probabilistic_finish_avoids_finishing_when_clearly_behind():
    from skyjo.strategies.heuristic import HeuristicStrategy, Profile

    s = HeuristicStrategy(Profile(name="t", finish_policy="probabilistic", card_counting=True))
    # Adversaire très bas, nous à 40 : finir coûte presque tout notre score.
    rnd = Round([s, RandomStrategy()], random.Random(0), first_player=0,
                boards=[[12, 11, 10, 12, 11, 10, 9, 9, 8, 1, 2, 3], [-2, -1, 0] * 4])
    for b in rnd.boards:
        b.reveal_all()
    rnd.boards[0].slot((3, 2)).visible = False
    view = rnd._view(0, False, 0)
    ctx = s._ctx(view)
    assert s._finish_value(view, ctx, 40) < -35


def test_strategy_overrides():
    s = make_strategy("stratege:finish_margin=0:hidden_max=none")
    assert s.p.finish_margin == 0 and s.p.hidden_max is None
    with pytest.raises(ValueError):
        make_strategy("stratege:inconnu=1")
