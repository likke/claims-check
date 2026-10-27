"""Mascot states map to the approved sprite frames, including severe claims."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mascot_js(expression, note=None, notes=None, pack="Tech claims"):
    script = f"""
global.window = global;
require('./static/mascot.js');
const note = {json.dumps(note)};
const notes = {json.dumps(notes or [])};
process.stdout.write(JSON.stringify({{
  frame: MascotStates.frameForState({json.dumps(expression)}),
  severe: MascotStates.isSevere(note, notes, {json.dumps(pack)})
}}));
"""
    result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_states_map_to_expected_mascot_frames():
    for state, frame in {
        "watching": "popcorn", "cur": "prop-magnifier", "worried": "prop-clipboard",
        "shocked": "prop-redcard", "unsure": "confused", "happy": "cool", "done": "party",
    }.items():
        assert mascot_js(state)["frame"] == frame


def test_severe_rule_and_cure_treat_claims_trigger_hide_sequence():
    rule = {"claim": {"rounds": [{"judgement": "rule"}]}}
    cure = {"claim": {"quote": "This cream cures eczema"}}
    assert mascot_js("watching", rule, [rule], "FDA beauty")["severe"]
    assert mascot_js("watching", cure, [cure])["severe"]


def test_two_failed_claims_within_ten_seconds_are_severe():
    a = {"claim": {"status": "fail", "start": 2}}
    b = {"claim": {"status": "fail", "start": 11.5}}
    assert mascot_js("watching", a, [a, b])["severe"]
    b["claim"]["start"] = 12.1
    assert not mascot_js("watching", a, [a, b])["severe"]
