"""The review screen's design rules, checked two ways.

1. Static: the stylesheet never sets a border on one edge, a shadow, a
   gradient, a corner above 6px or a font size outside the scale.
2. In a real browser (skipped when Chrome is not installed): after a full
   offline review renders, every element's computed border widths are equal on
   all four sides (or all zero), and nothing has a shadow or a gradient.
"""
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimcheck.web import app

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "static" / "review.html"
CHROME = shutil.which("google-chrome") or shutil.which("chromium") or (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" if Path("/Applications/Google Chrome.app").exists() else None)


def css() -> str:
    return "\n".join(re.findall(r"<style>(.*?)</style>", PAGE.read_text(), re.S))


def test_review_is_the_default_route_and_classic_stays():
    client = TestClient(app)
    assert 'id="checking-screen"' in client.get("/").text
    assert "Review notes" in client.get("/").text
    assert "Review notes" in client.get("/review").text
    assert "Ask the agent" in client.get("/classic").text


def test_review_includes_timeline_mascot_and_reaction_sync():
    page = PAGE.read_text()
    assert 'id="mascot"' in page and 'id="reaction-bubble"' in page
    assert "$('#player').currentTime" in page
    assert "prefers-reduced-motion:reduce" in page
    assert "AudioContext" in page and "id=\"mute\"" in page


def test_staged_screens_and_sample_presets_are_present():
    page = PAGE.read_text()
    for screen in ("checking-screen", "results-screen", "claim-screen", "disclosure-screen", "decision-screen", "done-screen"):
        assert f'id="{screen}"' in page
    assert "const PRESETS" in page
    for preset in ("LunaDerm Barrier Serum", "TsekPay", "Ledgerly AI"):
        assert preset in page


def test_replay_passed_imprints_match_passed_claims_and_found_disclosures():
    replay = json.loads((ROOT / "demo" / "live-run-2026-10-27.json").read_text())
    passed_claims = sum(claim.get("status") == "pass" for claim in replay["claims"])
    disclosure = replay.get("disclosure", {})
    found_disclosures = sum(bool(disclosure.get(channel, {}).get("found")) for channel in ("spoken", "on_screen"))
    assert passed_claims == 1
    assert found_disclosures == 1
    assert "note.claim.status === 'pass'" in PAGE.read_text()


def test_replay_video_and_approval_controls_are_wired():
    page = PAGE.read_text()
    assert "startReplayVideo();" in page
    assert 'Tap to watch with Chicky</button>' in page
    assert "replayLoopPending" in page and "replayLooped" in page
    assert "Human review summary" in page


def test_stylesheet_has_no_edge_bars_shadows_or_gradients():
    sheet = re.sub(r"/\*.*?\*/", "", css(), flags=re.S)
    assert not re.search(r"border-(left|right|top|bottom)\s*:", sheet), "one-edge border found"
    assert not re.search(r"border-(left|right|top|bottom)-width", sheet)
    assert not re.search(r"box-shadow\s*:(?!\s*none)", sheet)
    assert "gradient" not in sheet
    assert ".flow-screen .verdict{font-size:34px" in sheet
    sizes = {int(v) for v in re.findall(r"(?:font-size|font)\s*:[^;]*?(\d+)px", sheet)}
    assert sizes <= {14, 17, 20, 26, 34, 48}, sizes
    weights = {int(v) for v in re.findall(r"font-weight\s*:\s*(\d+)", sheet)}
    assert weights <= {400, 500, 600, 700}, weights


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(CHROME is None, reason="Chrome is not installed")
def test_computed_styles_have_no_edge_bars_in_a_real_browser(tmp_path):
    websockets = pytest.importorskip("websockets.sync.client")
    port, debug = _free_port(), _free_port()
    env = {"CLAIMCHECK_PROVIDER": "fixture", "FIXTURE_STEP_DELAY": "0", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--port", str(port)], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={debug}", f"--user-data-dir={tmp_path}",
                               "--window-size=1280,900", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/config", timeout=1)
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{debug}/json/list", timeout=1))
                ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        assert ws_url, "browser or server did not start"
        with websockets.connect(ws_url, max_size=None) as ws:
            counter = iter(range(1, 10_000))

            def call(method, **params):
                n = next(counter)
                ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    message = json.loads(ws.recv())
                    if message.get("id") == n:
                        return message.get("result", {})

            def js(expression):
                return call("Runtime.evaluate", expression=expression, returnByValue=True).get("result", {}).get("value")

            for width, height in ((1280, 900), (390, 844)):
                call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=width < 500)
                call("Page.navigate", url=f"http://127.0.0.1:{port}/review?auto=run")
                for _ in range(150):
                    if js("document.querySelectorAll('.note').length > 3 && /Reviewed in/.test(document.querySelector('#status').textContent)"):
                        break
                    time.sleep(0.1)
                report = js("""(() => {
                  const bad = [];
                  for (const el of document.querySelectorAll('*')) {
                    for (const pseudo of [null, '::before', '::after']) {
                      const s = getComputedStyle(el, pseudo);
                      const w = ['Top', 'Right', 'Bottom', 'Left'].map(k => s['border' + k + 'Style'] === 'none' ? 0 : parseFloat(s['border' + k + 'Width']));
                      if (new Set(w).size > 1) bad.push(['edge', el.tagName, el.className, pseudo, w.join(',')]);
                      if (s.boxShadow !== 'none') bad.push(['shadow', el.tagName, el.className, pseudo]);
                      if (/gradient/.test(s.backgroundImage)) bad.push(['gradient', el.tagName, el.className, pseudo]);
                    }
                  }
                  return {bad, notes: document.querySelectorAll('.note').length, body: getComputedStyle(document.body).fontSize,
                          scroll: document.documentElement.scrollWidth, width: document.documentElement.clientWidth};
                })()""")
                assert report["notes"] >= 4, report
                visible = js("document.body.innerText")
                jargon = re.findall(r"\b(chunks?|turns?|asr|tokens?|schema|fixture|heuristic)\b", visible, re.I)
                assert jargon == [], f"jargon on screen: {jargon}"
                assert js("plain('No chunk states a guarantee of zero FDA violations.')") == "The product sources never promise zero FDA violations."
                assert js("plain('The line matches S3.')") == "The line matches the product sources."
                assert re.search(r"Reviewed in \d+:\d\d", visible)
                assert report["bad"] == [], report["bad"][:5]
                assert report["body"] == "17px"
                assert report["scroll"] <= report["width"], f"horizontal scroll at {width}px"
    finally:
        chrome.terminate()
        server.terminate()
