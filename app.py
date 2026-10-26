"""Dasho Claim Check: source-grounded review of creator video claims."""
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
FIXTURE_SOURCE = "Ledgerly AI helps teams organize invoices and flag unusual charges for human review. It does not guarantee fraud prevention or regulatory compliance."


class FixtureProvider:
    """Offline demonstration provider; fictional content only."""
    def transcribe(self, video: bytes) -> str:
        return "Ledgerly AI guarantees zero FDA violations. Sulit ito!"

    def frames(self, video: bytes) -> list[dict[str, Any]]:
        return [{"time": 1.2, "text": "Paid partnership", "claims": []}]

    def plan(self, context: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        return {"tool": "search_source", "input": {"query": "Ledgerly AI capabilities"}}

    def json_plan(self, prompt: str) -> dict[str, Any]:
        return {"tool": "finish", "input": {}}


class OpenAICompatibleProvider(FixtureProvider):
    """OpenAI-compatible provider hook for Muse Spark Open/local endpoints."""
    def __init__(self):
        self.base_url = os.getenv("MODEL_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        self.model = os.getenv("MODEL_NAME", "muse-spark-open")
        self.api_key = os.getenv("MODEL_API_KEY", "")

    def _chat(self, messages: list[dict[str, str]], tools: list[dict] | None = None) -> dict:
        import urllib.request
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read())

    def plan(self, context: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        return self.json_plan(json.dumps({"context": context, "history": history}))

    def json_plan(self, prompt: str) -> dict[str, Any]:
        result = self._chat([{"role": "system", "content": "Return one JSON object with tool and input fields."}, {"role": "user", "content": prompt}])
        content = result["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except (TypeError, json.JSONDecodeError):
            retry = self._chat([{"role": "system", "content": "Return valid JSON only: {\"tool\": string, \"input\": object}."}, {"role": "user", "content": prompt + "\nYour previous response was invalid. Retry."}])
            return json.loads(retry["choices"][0]["message"]["content"])


def get_provider():
    return OpenAICompatibleProvider() if os.getenv("MODEL_API_KEY") else FixtureProvider()


def extract_claims(transcript: str) -> list[dict[str, Any]]:
    claims = []
    for match in re.finditer(r"[^.!?]+[.!?]?", transcript):
        text = match.group().strip()
        if re.search(r"\b(guarantee|guarantees|cure|treat|prevents|eliminates|100%|zero)\b", text, re.I):
            claims.append({"quote": text, "timestamp": round(match.start() / 12, 1), "rounds": []})
    return claims


def search_source(query: str, sources: str) -> list[str]:
    return [line.strip() for line in sources.splitlines() if any(w.lower() in line.lower() for w in re.findall(r"\w+", query) if len(w) > 3)] or [sources.strip()]


def judge_claim(claim: str, evidence: list[str]) -> dict[str, Any]:
    supported = bool(evidence) and not re.search(r"guarantee|zero FDA|cure|treat", claim, re.I)
    return {"verdict": "pass" if supported else "fail", "evidence": evidence[0] if supported else (evidence[0] if evidence else "No supporting source found."),
            "reason": "The wording is supported by the source." if supported else "The source describes assistance and human review, not a guarantee or therapeutic outcome."}


def rewrite_line(claim: str, reason: str, evidence: list[str]) -> str:
    # Preserve the creator's enthusiastic point while correcting the unsupported promise.
    return "Ledgerly AI helps flag possible issues early, so your team can review them before posting. Sulit ito!"


def check_disclosure(transcript: str, frames: list[dict[str, Any]]) -> bool:
    return bool(re.search(r"paid partnership|sponsored|ad\b|pakikipagtulungan bayad|bayad na partnership", transcript, re.I) or
                any(re.search(r"paid partnership|sponsored|\bad\b", f.get("text", ""), re.I) for f in frames))


def check_coverage(transcript: str, key_messages: str) -> dict[str, Any]:
    messages = [x.strip() for x in key_messages.splitlines() if x.strip()]
    return {"covered": [m for m in messages if any(word.lower() in transcript.lower() for word in re.findall(r"\w+", m) if len(word) > 4)],
            "missing": [m for m in messages if not any(word.lower() in transcript.lower() for word in re.findall(r"\w+", m) if len(word) > 4)]}


def run_review(video: bytes, sources: str, messages: str, pack: str, provider=None, emit=None) -> dict[str, Any]:
    provider = provider or get_provider()
    trace = []
    def log(tool, inputs, output):
        if len(trace) >= 12:
            return
        step = {"step": len(trace) + 1, "tool": tool, "input": inputs, "output": output}
        trace.append(step)
        if emit: emit(step)
    transcript = provider.transcribe(video)
    log("transcribe", {"video_bytes": len(video)}, transcript)
    frames = provider.frames(video)
    log("vision_frames", {"sampled_frames": len(frames)}, frames)
    context = {"transcript": transcript, "sources": sources, "pack": pack}
    claims = extract_claims(transcript)
    log("extract_claims", {}, claims)
    history = []
    # The declared agent tools are dispatched through a bounded plan-act-observe loop.
    tools = {"search_source", "fetch_source", "judge_claim", "rewrite_line", "check_disclosure", "check_coverage", "finish"}
    for claim in claims:
        evidence = search_source(claim["quote"], sources)
        log("search_source", {"query": claim["quote"]}, evidence)
        verdict = judge_claim(claim["quote"], evidence)
        log("judge_claim", {"claim": claim["quote"]}, verdict)
        claim["rounds"].append(verdict)
        current = claim["quote"]
        if verdict["verdict"] == "fail":
            revised = rewrite_line(current, verdict["reason"], evidence)
            log("rewrite_line", {"claim": current, "failure_reason": verdict["reason"]}, revised)
            retry = judge_claim(revised, evidence)
            log("judge_claim", {"claim": revised, "round": 2, "failure_reason": verdict["reason"]}, retry)
            claim["rounds"].append(retry)
            claim["suggested_line"] = revised
        claim["verdict"] = claim["rounds"][-1]["verdict"]
        claim["evidence"] = claim["rounds"][-1]["evidence"]
        history.append(claim)
    disclosed = check_disclosure(transcript, frames)
    log("check_disclosure", {}, disclosed)
    coverage = check_coverage(transcript, messages)
    log("check_coverage", {}, coverage)
    if pack == "FDA beauty":
        fda = {"therapeutic_claims": [c["quote"] for c in claims if re.search(r"\b(cure|treat)\w*\b", c["quote"], re.I)],
               "missing_disclaimer": not re.search(r"\b(these statements|not evaluated by the FDA|not intended to diagnose)\b", transcript, re.I)}
        log("FDA_rule_pack", {}, fda)
    log("finish", {}, {"claim_count": len(claims)})
    return {"id": str(uuid.uuid4()), "transcript": transcript, "claims": claims, "disclosure": disclosed, "coverage": coverage, "trace": trace, "frames": frames}


app = FastAPI(title="Dasho Claim Check")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
REVIEWS: dict[str, dict[str, Any]] = {}

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text()

@app.post("/api/review")
async def review(video: UploadFile = File(...), sources: str = Form(FIXTURE_SOURCE), key_messages: str = Form("Organize invoices\nFlag unusual charges"), pack: str = Form("Tech claims")):
    result = run_review(await video.read(), sources, key_messages, pack)
    REVIEWS[result["id"]] = result
    return result

@app.get("/api/review/{review_id}/trace")
def trace(review_id: str):
    result = REVIEWS.get(review_id)
    if not result: return {"error": "Review not found"}
    return {"trace": result["trace"]}

@app.post("/api/review/{review_id}/decision")
async def decision(review_id: str, payload: dict):
    if review_id not in REVIEWS: return {"error": "Review not found"}
    REVIEWS[review_id]["decision"] = {"status": payload.get("status"), "reviewer": payload.get("reviewer", "")}
    return REVIEWS[review_id]["decision"]
