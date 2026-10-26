import app
from eval import run_eval
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

class LiveMock(app.FixtureProvider):
    def __init__(self): self.calls=[]
    def transcribe(self, video): return "We guarantee a cure."
    def extract_claims(self, transcription):
        return [{"quote": "We guarantee a cure.", "start": 2.25, "end": 3.4, "estimated": False, "rounds": []}]
    def frames(self, video): return [{"time":1,"text":"","image":"encoded"}]
    def plan(self, context, history): return {"tool":"search_source","input":{},"mode":"json"}
    def judge_claim(self, claim, evidence):
        self.calls.append(("judge",claim))
        return {"verdict":"fail" if claim=="We guarantee a cure." else "pass","reason":"round 1 failed","evidence":"source"}
    def rewrite_line(self, claim, reason, evidence, round_number=1):
        self.calls.append(("rewrite",reason,round_number)); return "We help you feel better with care."
    def check_disclosure(self, transcript, frames):
        self.calls.append(("disclosure_frames",frames)); return True

def test_native_tool_fallback(monkeypatch):
    provider=app.MetaProvider.__new__(app.MetaProvider)
    provider.mode="tools"
    def request(path,payload):
        if "tools" in payload: raise RuntimeError("HTTP 400 unsupported tools")
        return {"output_text":'{"tool":"finish","input":{}}'}
    monkeypatch.setattr(provider,"_request",request)
    assert provider.plan({},[])["mode"]=="json"

def test_round_two_reason_and_frame_disclosure():
    provider=LiveMock(); result=app.run_review(b"x","source","", "Tech claims",provider)
    assert ("rewrite","round 1 failed",1) in provider.calls
    assert result["claims"][0]["rounds"][1]["verdict"]=="pass"
    assert ("disclosure_frames",provider.frames(b"x")) in provider.calls
    assert result["claims"][0]["start"] == 2.25
    assert result["claims"][0]["end"] == 3.4
    assert result["claims"][0]["estimated"] is False

def test_meta_claim_extraction_uses_model_and_estimates_text_timing(monkeypatch):
    provider = app.MetaProvider.__new__(app.MetaProvider)
    provider.model = "muse-spark"
    prompts = []
    def json_call(prompt):
        prompts.append(prompt)
        return {"claims": [{"quote": "Claim.", "start": 0, "end": 2, "estimated": True}]}
    monkeypatch.setattr(provider, "_json_call", json_call)
    claims = provider.extract_claims({"text": "Claim.", "segments": [], "duration": 2})
    assert claims[0]["estimated"] is True
    assert claims[0]["start"] == 0 and claims[0]["end"] == 2
    assert '"start": 0.0' in prompts[0]

def test_trace_sse_and_polling():
    result=app.run_review(b"x","source","", "Tech claims",LiveMock()); app.REVIEWS[result["id"]]=result
    client=TestClient(app.app)
    assert len(client.get(f"/api/review/{result['id']}/trace").json()["trace"])==len(result["trace"])
    assert "data:" in client.get(f"/api/review/{result['id']}/events").text

def test_eval_sample_metrics(tmp_path):
    class EvalMock(app.FixtureProvider):
        def judge_claim(self,claim,evidence): return {"verdict":"fail" if "cures" in claim or "treats" in claim or "guarantees" in claim or "removes every" in claim else "pass"}
        def check_disclosure(self,transcript,frames): return any(x in transcript.lower() for x in ("sponsored","paid partnership","ad:"))
    out=tmp_path/"results.json"
    payload=run_eval.run("eval/sample.csv",out=out,provider=EvalMock())
    assert len(payload["results"])==8 and out.exists()
    assert "agreement" in payload["summary"][run_eval.LABELS[0]]
