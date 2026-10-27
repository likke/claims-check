"""Pure timeline sync and reaction fallback behavior used by the review page."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_timecode_to_reaction_sync_tracks_seek_pause_and_four_second_window():
    script = """
    global.window = {};
    require('./static/reactions.js');
    const pick = window.ReviewReactions.activeReactionAt;
    const notes = [{n: 1, time: 3, reaction: 'first'}, {n: 2, time: 8, reaction: 'next'}, {n: 3, time: null}];
    const result = [pick(notes, 2.9), pick(notes, 3), pick(notes, 6.99), pick(notes, 7), pick(notes, 9), pick(notes, 12)];
    process.stdout.write(JSON.stringify(result.map(n => n && n.reaction)));
    """
    result = subprocess.run([NODE, "-e", script], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout == '[null,"first","first",null,"next",null]'


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_reaction_fallback_template_is_specific_and_uses_model_field_when_present():
    script = """
    global.window = {};
    require('./static/reactions.js');
    const fallback = window.ReviewReactions.fallbackReaction;
    process.stdout.write(JSON.stringify([
      fallback({kind: 'verified'}),
      fallback({kind: 'claim', claim: {rounds: [{judgement: 'contradicted'}]}}),
      fallback({kind: 'claim', claim: {review: true}}),
      fallback({kind: 'verified', reaction: 'Muse Spark wrote this.'})
    ]));
    """
    result = subprocess.run([NODE, "-e", script], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout == '["That claim lines up with the sources!","Wait, let’s check this claim closely!","I’m not sure about this language. Let’s check together.","Muse Spark wrote this."]'
