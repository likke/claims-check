"""Post caption, on-screen claims, image and carousel posts, language checks and demo-mode safety."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from claimcheck import demo, fixture
from claimcheck.tools import ToolBox, detect_language
from helpers import Scripted, extraction, meta_with, policy, run, state, supported, responses_text

SOURCES = "Ledgerly AI flags unusual charges for a human reviewer.\nLedgerly AI does not guarantee fraud prevention."
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CLAIMCHECK_PROVIDER", "fixture")
    monkeypatch.setenv("FIXTURE_STEP_DELAY", "0")
    for name in ("DEMO_MODE", "DEMO_DAILY_LIMIT", "DEMO_MAX_MB", "DEMO_MAX_SECONDS", "DEMO_KEEP_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    demo.USAGE.clear()
    from claimcheck.web import app
    return TestClient(app)


def events(client, review_id):
    out = []
    with client.stream("GET", f"/api/reviews/{review_id}/events") as response:
        name = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                out.append((name, json.loads(line[6:])))
    return out


# ---- 1. post caption -------------------------------------------------------

def test_caption_claims_are_extracted_with_their_origin():
    caption = "Ledgerly AI guarantees zero fraud for every business! #ad"
    answer = extraction((0, "Ledgerly AI flags unusual charges.")) | {
        "caption_claims": [{"quote": "Ledgerly AI guarantees zero fraud for every business!"}, {"quote": "It doubles revenue."}]}
    st = state([{"start": 0, "end": 3, "text": "Ledgerly AI flags unusual charges."}], SOURCES)
    st.caption = caption
    provider = Scripted(responses={"extract_claims": answer})
    observation, _ = ToolBox(st, provider).run("extract_claims", {})
    spoken, from_caption = st.claims
    assert spoken.origin == "speech" and from_caption.origin == "caption" and from_caption.start is None
    assert [r["why"] for r in observation["rejected"]] == ["quote not found in the post caption"]
    assert "Post caption" in provider.prompts["extract_claims"][0] and caption in provider.prompts["extract_claims"][0]


@pytest.mark.parametrize("caption,found", [("Love this app #ad", True), ("May bayad na partnership ito with Ledgerly", True),
                                           ("Sponsored by Ledgerly AI", True), ("Just my honest review", False)])
def test_caption_disclosure_is_found_by_phrase(caption, found):
    st = state([{"start": 0, "end": 3, "text": "Ledgerly AI flags unusual charges."}], SOURCES)
    st.caption = caption
    ToolBox(st, Scripted(responses={"check_disclosure": {"spoken": {"found": False}, "on_screen": {"found": False}, "frame_text": []}})).run("check_disclosure", {})
    assert st.disclosure["caption"]["found"] is found and st.disclosure["disclosed"] is found


def test_model_caption_quote_must_be_in_the_caption():
    st = state([{"start": 0, "end": 3, "text": "Ledgerly AI flags unusual charges."}], SOURCES)
    st.caption = "Check out Ledgerly AI today."
    answer = {"spoken": {"found": False}, "on_screen": {"found": False}, "frame_text": [], "caption": {"found": True, "quote": "#sponsored"}}
    ToolBox(st, Scripted(responses={"check_disclosure": answer})).run("check_disclosure", {})
    assert st.disclosure["disclosed"] is False


def test_caption_claim_runs_through_judge_and_rewrite():
    caption = "Ledgerly AI guarantees zero fraud, promise!"
    responses = {
        "extract_claims": {"claims": [], "on_screen_claims": [], "caption_claims": [{"quote": caption}]},
        "judge_claim": lambda ctx, prompt: supported("S1", "flags unusual charges for a human reviewer") if "flags" in ctx["line"] else
            {"verdict": "contradicted", "chunk_id": "S2", "evidence_quote": "does not guarantee fraud prevention", "reason": "No guarantee."},
        "rewrite_line": {"line": "Ledgerly AI flags unusual charges against fraud, promise!", "chunk_ids": ["S1"]},
        "check_disclosure": {"spoken": {"found": False}, "on_screen": {"found": False}, "frame_text": []},
        "check_coverage": {"messages": []},
    }
    st = state([{"start": 0, "end": 2, "text": "Hello!"}], SOURCES)
    st.caption = caption
    provider = Scripted(policy, responses)
    result = run(provider, st)
    claim = result["claims"][0]
    assert claim["origin"] == "caption" and claim["start"] is None and claim["status"] == "fixed"
    assert "post caption sentence" in provider.prompts["rewrite_line"][0]


def test_caption_goes_through_the_api(client):
    response = client.post("/api/reviews", data={"sources": fixture.SOURCE_PACK, "key_messages": "x",
                                                 "caption": "Ledgerly AI connects with every bank in Asia. #ad"})
    result = events(client, response.json()["id"])[-1][1]
    origins = {c["origin"] for c in result["claims"]}
    assert "caption" in origins and result["inputs"]["caption"].endswith("#ad")
    assert result["disclosure"]["caption"]["found"] is True


# ---- 2. on-screen text claims ----------------------------------------------

def test_on_screen_claims_are_judged_like_spoken_ones():
    frames = [{"time": 1.0, "image": "aW1n"}, {"time": 2.5, "image": "aW1n"}]
    responses = {
        "extract_claims": {"claims": [], "on_screen_claims": [{"frame_time": 2.5, "text": "ZERO FRAUD 100%"}]},
        "judge_claim": {"verdict": "contradicted", "chunk_id": "S2", "evidence_quote": "does not guarantee fraud prevention", "reason": "No guarantee."},
        "rewrite_line": lambda ctx, prompt: {"line": "Flags unusual charges, fraud" if ctx["round"] == 1 else "Flags unusual charges for review, not zero fraud", "chunk_ids": ["S1"]},
        "check_disclosure": {"spoken": {"found": False}, "on_screen": {"found": False}, "frame_text": []},
        "check_coverage": {"messages": []},
    }
    provider = Scripted(policy, responses)
    result = run(provider, state([{"start": 0, "end": 3, "text": "Hi."}], SOURCES, frames=frames))
    claim = result["claims"][0]
    assert claim["origin"] == "on_screen" and claim["start"] == 2.5 and claim["rounds"][0]["judgement"] == "contradicted"
    assert "short on-screen text" in provider.prompts["rewrite_line"][0]
    assert len(provider.images["extract_claims"][0]) == 2


# ---- 3. image and carousel posts -------------------------------------------

def png(path: Path, size: str, colour: str):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s={size}", "-frames:v", "1", str(path)], check=True)
    return path.read_bytes()


@needs_ffmpeg
def test_carousel_post_has_numbered_images_and_no_timeline(client, tmp_path):
    files = [("images", ("one.png", png(tmp_path / "1.png", "1080x1350", "red"), "image/png")),
             ("images", ("two.jpg", png(tmp_path / "2.jpg", "1080x1080", "blue"), "image/jpeg"))]
    response = client.post("/api/reviews", data={"sources": fixture.SOURCE_PACK, "caption": "Sponsored. Ledgerly AI reads every invoice."}, files=files)
    assert response.status_code == 200, response.text
    stream = events(client, response.json()["id"])
    steps = [d for n, d in stream if n == "step"]
    info = next(s for s in steps if s.get("kind") == "media_info")
    images = next(s for s in steps if s.get("kind") == "post_images")["images"]
    assert info["layout"] == "layout-images" and info["images"] == 2
    assert [i["number"] for i in images] == [1.0, 2.0] and images[0]["height"] == 1024 and abs(images[0]["width"] - 819) <= 1
    result = stream[-1][1]
    assert result["post_kind"] == "images" and result["timing"] == "image post" and [f["time"] for f in result["frames"]] == [1.0, 2.0]
    assert result["trace"][0]["summary"].startswith("Image post with 2 images")


def test_image_prompts_name_images_not_seconds():
    from claimcheck import prompts
    from claimcheck.packs import get_pack
    text = prompts.extract_claims(get_pack("tech"), "X", [], [{"time": 1.0, "kind": "image"}, {"time": 2.0, "kind": "image"}])
    assert "image 1 (use frame_time 1), image 2 (use frame_time 2)" in text


@needs_ffmpeg
def test_image_upload_rules(client, tmp_path):
    good = png(tmp_path / "a.png", "64x64", "red")
    assert client.post("/api/reviews", data={"sources": "docs"}, files=[("images", ("a.gif", good, "image/gif"))]).status_code == 415
    many = [("images", (f"{i}.png", good, "image/png")) for i in range(11)]
    assert client.post("/api/reviews", data={"sources": "docs"}, files=many).status_code == 400
    both = [("images", ("a.png", good, "image/png")), ("video", ("v.mp4", b"0000", "video/mp4"))]
    assert client.post("/api/reviews", data={"sources": "docs"}, files=both).status_code == 400


# ---- 4. languages ----------------------------------------------------------

def test_taglish_is_verified_and_bisaya_is_flagged_for_a_person():
    taglish = state([{"start": 0, "end": 3, "text": "Grabe, sobrang bilis talaga ng Ledgerly AI."}], SOURCES)
    assert detect_language(taglish, "taglish")["verified"] is True
    bisaya = state([{"start": 0, "end": 3, "text": "Nindot kaayo ni nga app, dili jud ka magmahay."}], SOURCES)
    report = detect_language(bisaya, "")
    assert report["detected"] == "cebuano" and report["verified"] is False and "kaayo" in report["how"]
    assert detect_language(taglish, "cebuano")["verified"] is False  # the model's reading counts too


def test_unverified_language_marks_every_claim_for_human_review():
    turns = [{"start": 0, "end": 3, "text": "Nindot kaayo ni nga app, dili jud ka magmahay."},
             {"start": 3, "end": 6, "text": "Ledgerly AI guarantees zero fraud."}]
    provider = Scripted(responses={"extract_claims": extraction((1, turns[1]["text"])) | {"language": "cebuano"}})
    st = state(turns, SOURCES)
    observation, _ = ToolBox(st, provider).run("extract_claims", {})
    assert observation["language"]["verified"] is False and all(c.needs_human_review for c in st.claims)


def test_speech_bias_stays_tagalog_and_english():
    def handler(request):
        assert '"languageBias": ["Tagalog", "English"]' in request.content.decode("latin-1")
        return httpx.Response(200, json={"transcript": "Hi.", "audioDurationMs": 1000, "turns": [{"startMs": 0, "endMs": 1000, "transcript": "Hi."}]})
    assert meta_with(handler).transcribe(b"RIFF", [])["turns"]


# ---- 5. demo mode ----------------------------------------------------------

def test_demo_limits_live_checks_per_address(client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "2")
    data = {"sources": fixture.SOURCE_PACK}
    assert client.get("/api/config").json()["demo"]["checks_left"] == 2
    assert client.post("/api/reviews", data=data).status_code == 200
    assert client.post("/api/reviews", data=data).status_code == 200
    refused = client.post("/api/reviews", data=data)
    assert refused.status_code == 429 and "replay" in refused.json()["detail"]
    assert client.get("/api/replay").status_code == 200  # replay always works


@needs_ffmpeg
def test_demo_caps_size_and_length(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_MODE", "1")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:d=3:r=10", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(tmp_path / "c.mp4")], check=True)
    clip = (tmp_path / "c.mp4").read_bytes()
    monkeypatch.setenv("DEMO_MAX_SECONDS", "2")
    too_long = client.post("/api/reviews", data={"sources": "docs"}, files={"video": ("c.mp4", clip, "video/mp4")})
    assert too_long.status_code == 413 and "seconds" in too_long.json()["detail"]
    monkeypatch.setenv("DEMO_MAX_SECONDS", "60")
    monkeypatch.setenv("DEMO_MAX_MB", "0")
    assert client.post("/api/reviews", data={"sources": "docs"}, files={"video": ("c.mp4", clip, "video/mp4")}).status_code == 413


@needs_ffmpeg
def test_demo_deletes_uploads_after_the_check(client, monkeypatch, tmp_path):
    from claimcheck.web import JOBS
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_KEEP_SECONDS", "0")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:d=2:r=10", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(tmp_path / "c.mp4")], check=True)
    response = client.post("/api/reviews", data={"sources": fixture.SOURCE_PACK}, files={"video": ("c.mp4", (tmp_path / "c.mp4").read_bytes(), "video/mp4")})
    events(client, response.json()["id"])
    job = JOBS[response.json()["id"]]
    for _ in range(50):
        if not job.folder.exists():
            break
        time.sleep(0.05)
    assert job.folder is not None and not job.folder.exists()


def test_the_key_never_reaches_the_browser(client, monkeypatch):
    secret = "sk-test-NEVER-SHOW-THIS-0123456789"
    monkeypatch.setenv("MODEL_API_KEY", secret)
    monkeypatch.setenv("DEMO_MODE", "1")
    bodies = [client.get("/api/config").text, client.get("/api/replay").text, client.get("/").text]
    review_id = client.post("/api/reviews", data={"sources": fixture.SOURCE_PACK}).json()["id"]
    bodies.append(json.dumps(events(client, review_id)))
    bodies.append(client.get(f"/api/reviews/{review_id}").text)
    bodies.append(client.post("/api/ask", json={"review_id": review_id, "question": "What failed?"}).text)
    assert all(secret not in body for body in bodies)
