from skyjo.simulate import run_match, wilson_interval
from skyjo.sweep import run_sweep
from skyjo.tournament import run_tournament, tables_for


def test_match_is_reproducible_and_counts_add_up():
    specs = ["prudent", "fonceur", "collectionneur", "opportuniste"]
    a = run_match(specs, 20, seed=7, log_games=2)
    b = run_match(specs, 20, seed=7)
    assert [r["win_rate"] for r in a.summaries()] == [r["win_rate"] for r in b.summaries()]
    assert sum(s.wins for s in a.stats.values()) == 20
    assert all(s.appearances == 20 for s in a.stats.values())
    assert {d["game"] for d in a.decisions} == {0, 1}


def test_duplicate_strategies_are_merged():
    res = run_match(["greedy", "greedy", "random"], 10, seed=1)
    assert res.stats["greedy"].appearances == 20


def test_parallel_chunks_match_total_games():
    res = run_match(["greedy", "fonceur"], 230, seed=1, jobs=2, log_games=150)
    assert res.n_games == 230
    assert {d["game"] for d in res.decisions} == set(range(150))


def test_results_do_not_depend_on_jobs():
    a = run_match(["greedy", "fonceur"], 210, seed=3, jobs=1).summaries()
    b = run_match(["greedy", "fonceur"], 210, seed=3, jobs=3).summaries()
    assert a == b


def test_progress_reaches_total():
    import io

    from skyjo.progress import Progress

    stream = io.StringIO()
    with Progress(250, label="t", groups=2, stream=stream) as pr:
        run_tournament(["prudent", "fonceur"], [2], 250, seed=1, progress=pr)
    assert pr.done == 250 and pr.groups_done == 1
    assert "100.0%" in stream.getvalue()


def test_tables_and_tournament():
    assert len(tables_for(["a", "b", "c", "d", "e"], 4)) == 5
    res = run_tournament(["prudent", "fonceur", "stratege"], [2, 3], 5, seed=1)
    assert set(res.by_players) == {2, 3}
    assert len(res.matches) == 3 + 1


def test_wilson_bounds():
    lo, hi = wilson_interval(0.5, 100)
    assert 0.39 < lo < 0.5 < hi < 0.61


def test_sweep_returns_one_row_per_value():
    rows = run_sweep("prudent", "hidden_max", ["2", "4"], ["fonceur"], 5, seed=1)
    assert [r["value"] for r in rows] == ["2", "4"]
    assert rows[0]["strategy"] == "prudent[hidden_max=2]"
