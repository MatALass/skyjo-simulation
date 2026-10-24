import random

import pytest

from skyjo.board import Board, BoardView, Slot
from skyjo.ml.features import FEATURES, apply_place, featurize, grid_from_view, make_context

pytest.importorskip("sklearn")


def test_grid_from_view_hides_hidden_cards():
    board = Board.from_columns([[Slot(5, True), Slot(9, False), Slot(1, True)], None,
                                [Slot(2, False)] * 3, [Slot(0, True)] * 3])
    assert grid_from_view(BoardView(board)) == [[5, None, 1], None, [None, None, None], [0, 0, 0]]


def test_apply_place_removes_completed_column():
    grid = [[7, 7, None], [1, 2, 3], None, [None] * 3]
    after = apply_place(grid, (0, 2), 7)
    assert after[0] is None and grid[0] == [7, 7, None]  # copie, l'original est intact


def test_featurize_counts():
    ctx = make_context([[[1, 2, 3], [None] * 3, None, [4, None, 4]]], e_unknown=5.0, is_last_turn=False, turn=3)
    f = dict(zip(FEATURES, featurize([[9, 9, None], [0, 0, 1], None, [None] * 3], ctx), strict=True))
    assert len(f) == len(FEATURES)
    assert f["my_visible"] == 19 and f["my_hidden"] == 4 and f["my_est"] == 39
    assert f["open_pairs"] == 1 and f["open_pair_value"] == 9
    assert f["closed_pairs"] == 1 and f["closed_pair_value"] == 0
    assert f["cols_removed"] == 1 and f["cols_hidden_3"] == 1
    assert f["opp_best_est"] == 6 + 8 + 4 * 5.0 and f["i_finish"] == 0


def test_end_to_end_dataset_train_play(tmp_path):
    from skyjo.ml.dataset import generate
    from skyjo.ml.train import train
    from skyjo.simulate import run_match

    df = generate(30, seed=1, pool=("expert", "prudent"), eps=0.2)
    assert len(df) > 1000 and df["round_score"].notna().all()
    assert df.groupby(["game", "round"])["round_score"].nunique().max() <= 4  # une cible par joueur et manche
    model = tmp_path / "value.joblib"
    res = train(df, model, max_iter=20, importance_rows=500)
    assert res.metrics["gradient_boosting"]["mae"] > 0 and model.exists()
    match = run_match([f"ml:{model}", "expert"], 3, seed=2)
    assert sum(s.appearances for s in match.stats.values()) == 6
    random.seed(0)


def test_model_loading_limits_openmp_threads(tmp_path):
    """Sans cette limite, les simulations parallèles deviennent 40x plus lentes (sursouscription)."""
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

    from skyjo.ml.dataset import generate
    from skyjo.ml.policy import load_model
    from skyjo.ml.train import train

    model = tmp_path / "m.joblib"
    train(generate(5, seed=3, pool=("expert",)), model, max_iter=5, importance_rows=200)
    load_model(str(model))
    assert _openmp_effective_n_threads() == 1
