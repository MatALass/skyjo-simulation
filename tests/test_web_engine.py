import json
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import pytest

from skyjo.assist import advise, build_assistant, parity_cases, profiles_json
from skyjo.coach import parse_grid

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js absent : parité JS non vérifiable")
def test_js_engine_matches_python(tmp_path):
    cases = parity_cases(n=250, seed=1)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    (tmp_path / "profiles.json").write_text(json.dumps(profiles_json()))
    engine = resources.files("skyjo.web").joinpath("engine.js")
    runner = Path(__file__).with_name("web_parity.js")
    args = [NODE, str(runner), str(engine), str(tmp_path / "cases.json"), str(tmp_path / "profiles.json")]
    res = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr


def test_advise_start_of_turn_and_drawn():
    me = parse_grid("5 ? ? 3 / ? 12 ? ? / 0 ? ? ?")
    opp = [parse_grid("? 4 ? ? / ? ? ? 1 / ? ? 2 ?")]
    start = advise(me, opp, discard=-2)
    assert start["step"] == "discard_choice" and start["take"] is True
    assert start["placement"]["chosen"]["total"] >= 5
    drawn = advise(me, opp, discard=7, drawn=11)
    assert drawn["step"] == "drawn" and drawn["options"]


def test_build_assistant_embeds_engine_and_profiles():
    page = build_assistant()
    assert page.startswith("<!doctype html>")
    assert "/*__ENGINE__*/" not in page and "/*__PROFILES__*/" not in page
    assert "class Heuristic" in page and '"expert"' in page
