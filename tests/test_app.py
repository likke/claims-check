import app
from fastapi.testclient import TestClient

class MockProvider(app.FixtureProvider):
    def __init__(self, transcript="Ledgerly AI guarantees zero FDA violations. Sulit ito!", frames=None):
        self.text = transcript
        self.frame_data = frames or [{"time": 1, "text": "Paid partnership", "claims": []}]
    def transcribe(self, video): return self.text
    def frames(self, video): return self.frame_data

def test_claim_extraction_and_search():
    assert len(app.extract_claims("We guarantee success.")) == 1
    assert app.search_source("organize invoices", "Ledgerly helps organize invoices.")[0].startswith("Ledgerly")

def test_judge_two_round_rewrite():
    evidence = [app.FIXTURE_SOURCE]
    first = app.judge_claim("guarantees zero FDA violations", evidence)
    revised = app.rewrite_line("guarantees zero FDA violations", first["reason"], evidence)
    second = app.judge_claim(revised, evidence)
    assert first["verdict"] == "fail" and second["verdict"] == "pass"
    assert "Ledgerly AI" in revised and "review" in revised

def test_taglish_disclosure():
    assert app.check_disclosure("May paid partnership kami, sulit!", [])

def test_fda_pack_and_endpoint():
    result = app.run_review(b"video", app.FIXTURE_SOURCE, "Organize invoices", "FDA beauty", MockProvider())
    assert any(t["tool"] == "FDA_rule_pack" for t in result["trace"])
    client = TestClient(app.app)
    response = client.post("/api/review", files={"video": ("clip.mp4", b"video", "video/mp4")}, data={"pack": "FDA beauty"})
    assert response.status_code == 200
    rid = response.json()["id"]
    assert client.get(f"/api/review/{rid}/trace").json()["trace"]
    assert client.post(f"/api/review/{rid}/decision", json={"status":"approved", "reviewer":"Ari"}).json()["reviewer"] == "Ari"

def test_step_cap():
    result = app.run_review(b"x", app.FIXTURE_SOURCE, "", "Tech claims", MockProvider("We guarantee A. We guarantee B. We guarantee C."))
    assert len(result["trace"]) <= 12

def test_json_retry(monkeypatch):
    provider = app.OpenAICompatibleProvider()
    values = iter([{"choices":[{"message":{"content":"bad json"}}]}, {"choices":[{"message":{"content":"{\"tool\":\"finish\",\"input\":{}}"}}]}])
    monkeypatch.setattr(provider, "_chat", lambda *a, **k: next(values))
    assert provider.json_plan("plan")["tool"] == "finish"
