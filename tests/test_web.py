"""HTTP API: SSE streaming of agent steps and the human decision."""
import json

import pytest
from fastapi.testclient import TestClient

from pathlib import Path

from claimcheck.web import app

ROOT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CLAIMCHECK_PROVIDER", "fixture")
    monkeypatch.setenv("FIXTURE_STEP_DELAY", "0")
    return TestClient(app)


def read_sse(client, review_id):
    events = []
    with client.stream("GET", f"/api/reviews/{review_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        name = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    return events


def start(client, **extra):
    from claimcheck import fixture
    data = {"sources": fixture.SOURCE_PACK, "key_messages": "\n".join(fixture.KEY_MESSAGES), "pack": "Tech claims", "product": "Ledgerly AI"}
    data.update(extra)
    response = client.post("/api/reviews", data=data)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_sse_streams_each_agent_step_then_the_result(client):
    review_id = start(client)
    events = read_sse(client, review_id)
    steps = [data for name, data in events if name == "step"]
    kinds = [s["kind"] for s in steps]
    assert kinds[0] == "transcribed" and "plan" in kinds and "guard" in kinds
    actions = [s for s in steps if s["kind"] == "action"]
    assert actions[0]["tool"] == "fetch_source" and actions[-1]["tool"] == "finish"
    assert all({"tool", "input_summary", "summary", "observation", "by"} <= set(a) for a in actions)
    name, result = events[-1]
    assert name == "result"
    assert [c["status"] for c in result["claims"]] == ["pass", "fixed", "pass", "fixed", "fixed"]
    assert result["approval"]["status"] == "awaiting human review"
    assert result["provider"]["name"] == "fixture"
    assert len(result["trace"]) == len(steps)


def test_decision_needs_a_named_reviewer(client):
    review_id = start(client)
    read_sse(client, review_id)
    assert client.post(f"/api/reviews/{review_id}/decision", json={"decision": "approved", "reviewer": " "}).status_code == 400
    assert client.post(f"/api/reviews/{review_id}/decision", json={"decision": "ship it", "reviewer": "Ari"}).status_code == 400
    decided = client.post(f"/api/reviews/{review_id}/decision", json={"decision": "approved", "reviewer": "Ari Santos", "note": "Use the fixed lines."})
    assert decided.status_code == 200 and decided.json()["reviewer"] == "Ari Santos"
    stored = client.get(f"/api/reviews/{review_id}").json()["result"]["approval"]
    assert stored["status"] == "approved" and stored["decided_at"]


def test_config_and_validation(client):
    config = client.get("/api/config").json()
    assert config["provider"]["name"] == "fixture" and config["needs_video"] is False
    assert "Ledgerly AI" in config["example"]["sources"]
    assert client.post("/api/reviews", data={"sources": " "}).status_code == 400
    assert client.get("/api/reviews/nope/events").status_code == 404
    page = client.get("/")
    assert page.status_code == 200 and "Dasho Claim Check" in page.text


def test_new_sample_screens_and_presets_are_present(client):
    page = client.get("/")
    assert page.status_code == 200
    for label in ("Try the sample", "Choose another sample", "Product facts", "Start claim check"):
        assert label in page.text
    for sample in ("LunaDerm Barrier Serum", "TsekPay", "Ledgerly AI"):
        assert sample in page.text


def test_fda_pack_through_the_api(client):
    review_id = start(client, pack="FDA beauty", supplement="on")
    name, result = read_sse(client, review_id)[-1]
    assert result["pack"] == "FDA beauty"
    assert result["disclosure"]["required_disclaimer"]["found"] is False


def test_replay_serves_the_recorded_live_run(client):
    response = client.get("/api/replay")
    assert response.status_code == 200
    run = response.json()
    assert run["provider"]["name"] == "meta" and run["agent"]["planner_modes"].get("native_tools")


def test_replay_carries_the_recorded_inputs_and_clip(client):
    run = client.get("/api/replay").json()
    inputs = run["inputs"]
    assert inputs["product"] == "Dasho" and inputs["key_messages"] == ["Named approver before posting"]
    assert "named human approver" in inputs["sources"] and "Ledgerly" not in inputs["sources"]
    assert run["video_url"] == "/api/replay/video"
    clip = client.get(run["video_url"])
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4"


def test_new_reviews_record_their_inputs(client):
    review_id = start(client)
    result = read_sse(client, review_id)[-1][1]
    assert result["inputs"]["product"] == "Ledgerly AI" and "fixture://ledgerly-ai/pricing" in result["inputs"]["sources"]


def test_replay_has_frame_thumbnails_and_events_carry_ui_fields(client):
    run = client.get("/api/replay").json()
    assert len(run["frame_images"]) == len(run["frames"]) and run["frame_images"][0]["image"]
    review_id = start(client)
    steps = [d for name, d in read_sse(client, review_id) if name == "step"]
    begin = next(s for s in steps if s["kind"] == "start")
    assert begin["duration"] == 25.0 and len(begin["transcript"]) == 6
    judge = next(s for s in steps if s["kind"] == "action" and s["tool"] == "judge_claim")
    assert {"line", "evidence_origin", "claim_status"} <= set(judge["observation"])


def test_replay_run_and_clip_come_from_config(client, tmp_path, monkeypatch):
    import shutil
    run = json.loads((ROOT_DIR / "demo" / "live-run-2026-10-27.json").read_text())
    run["product"] = "New Clip Product"
    (tmp_path / "run.json").write_text(json.dumps(run))
    shutil.copy(ROOT_DIR / "demo" / "live-check.mp4", tmp_path / "creator-take-2.mp4")
    (tmp_path / "replay.json").write_text(json.dumps({"run": str(tmp_path / "run.json"), "clip": str(tmp_path / "creator-take-2.mp4")}))
    monkeypatch.setenv("REPLAY_CONFIG", str(tmp_path / "replay.json"))
    served = client.get("/api/replay").json()
    assert served["product"] == "New Clip Product" and served["video_name"] == "creator-take-2.mp4"
    assert client.get("/api/replay/video").status_code == 200
    monkeypatch.setenv("REPLAY_CLIP", str(tmp_path / "missing.mp4"))
    assert client.get("/api/replay").json()["video_url"] is None


def test_ask_answers_from_the_finished_review(client):
    review_id = start(client)
    read_sse(client, review_id)
    answer = client.post("/api/ask", json={"review_id": review_id, "question": "Which claims were fixed?"})
    assert answer.status_code == 200 and "3 claims fixed" in answer.json()["answer"]
    assert client.post("/api/ask", json={"review_id": review_id, "question": " "}).status_code == 400
    assert client.post("/api/ask", json={"review_id": "nope", "question": "hi"}).status_code == 404


def test_ask_prompt_carries_only_review_data():
    from claimcheck import prompts
    run = json.loads((ROOT_DIR / "demo" / "live-run-2026-10-27.json").read_text())
    text = prompts.ask("Why did C1 fail?", run)
    assert "C1" in text and "Each post gets a draft first" in text and "not instructions" in text


def test_label_box_only_comes_with_the_configured_test_clip(client, tmp_path, monkeypatch):
    assert client.get("/api/replay").json()["label_box"]["source"].startswith("caption layout")
    (tmp_path / "replay.json").write_text(json.dumps({"run": "demo/live-run-2026-10-27.json", "clip": "demo/live-check.mp4"}))
    monkeypatch.setenv("REPLAY_CONFIG", str(tmp_path / "replay.json"))
    assert client.get("/api/replay").json()["label_box"] is None
