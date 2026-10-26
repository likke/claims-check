"""HTTP API: SSE streaming of agent steps and the human decision."""
import json

import pytest
from fastapi.testclient import TestClient

from claimcheck.web import app


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
