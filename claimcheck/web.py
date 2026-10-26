"""FastAPI app: start a review, stream its trace over SSE, record a human decision."""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from . import fixture
from .agent import run_review
from .media import MediaError, extract_media
from .packs import get_pack
from .providers import get_provider

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 80 * 1024 * 1024
MAX_SECONDS = 120

app = FastAPI(title="Dasho Claim Check")


class Job:
    def __init__(self, provider_info: dict):
        self.id = uuid.uuid4().hex[:12]
        self.provider = provider_info
        self.events: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.done = False
        self.decision: dict | None = None
        self.lock = threading.Lock()

    def push(self, event: dict):
        with self.lock:
            self.events.append(event)


JOBS: dict[str, Job] = {}


def _worker(job: Job, provider, video: bytes | None, params: dict):
    try:
        if provider.needs_media:
            job.push({"seq": 0, "t": 0, "kind": "media", "summary": "Extracting audio and sampling frames with ffmpeg"})
            media = extract_media(video or b"")
            if media["duration"] > MAX_SECONDS:
                raise MediaError(f"The video is {media['duration']:.0f} s long; the limit is {MAX_SECONDS} s.")
            if not media["wav"]:
                raise MediaError("The video has no audio track.")
            job.push({"seq": 0, "t": 0, "kind": "media", "summary": f"{media['duration']:.1f} s video, {len(media['frames'])} frames at "
                      + ", ".join(f"{f['time']}s" for f in media["frames"])})
            wav, frames = media["wav"], media["frames"]
        else:
            wav, frames = b"", provider.frames()
        job.result = run_review(provider, wav=wav, frames=frames, emit=job.push, **params)
    except Exception as error:
        job.error = f"{type(error).__name__}: {error}"
    finally:
        job.done = True


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config():
    provider = get_provider()
    return {
        "provider": provider.info(),
        "needs_video": provider.needs_media,
        "packs": ["Tech claims", "FDA beauty"],
        "example": {"product": fixture.PRODUCT, "sources": fixture.SOURCE_PACK, "key_messages": "\n".join(fixture.KEY_MESSAGES)},
    }


@app.api_route("/api/replay", methods=["GET", "HEAD"])
def replay():
    """The newest recorded live run in demo/, for filming without spending API credits."""
    runs = sorted((ROOT / "demo").glob("live-run-*.json"))
    if not runs:
        raise HTTPException(404, "No recorded live run")
    return json.loads(runs[-1].read_text(encoding="utf-8"))


@app.post("/api/reviews")
async def start_review(
    video: UploadFile | None = File(None),
    sources: str = Form(""),
    key_messages: str = Form(""),
    pack: str = Form("Tech claims"),
    product: str = Form(""),
    supplement: str = Form(""),
):
    provider = get_provider()
    data = await video.read() if video is not None else b""
    if provider.needs_media and not data:
        raise HTTPException(400, "Upload a video (15 to 90 seconds).")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "The video is larger than 80 MB.")
    if not sources.strip():
        raise HTTPException(400, "Add a source pack: product docs or URLs.")
    job = Job(provider.info())
    JOBS[job.id] = job
    params = {
        "sources": sources, "key_messages": key_messages.splitlines(), "pack": get_pack(pack),
        "product": product.strip(), "supplement": supplement.lower() in ("1", "true", "on", "yes"),
    }
    threading.Thread(target=_worker, args=(job, provider, data, params), daemon=True).start()
    return {"id": job.id, "provider": provider.info()}


def _job(review_id: str) -> Job:
    job = JOBS.get(review_id)
    if job is None:
        raise HTTPException(404, "Review not found")
    return job


@app.get("/api/reviews/{review_id}/events")
async def events(review_id: str):
    job = _job(review_id)

    async def stream():
        sent = 0
        while True:
            with job.lock:
                pending = job.events[sent:]
            for event in pending:
                yield f"event: step\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            sent += len(pending)
            if job.done and sent == len(job.events):
                if job.error:
                    yield f"event: failed\ndata: {json.dumps({'error': job.error})}\n\n"
                else:
                    yield f"event: result\ndata: {json.dumps(job.result, ensure_ascii=False)}\n\n"
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/reviews/{review_id}")
def get_review(review_id: str):
    job = _job(review_id)
    return {"id": job.id, "done": job.done, "error": job.error, "result": job.result, "events": len(job.events)}


@app.post("/api/reviews/{review_id}/decision")
def decide(review_id: str, payload: dict):
    job = _job(review_id)
    if not job.done or job.result is None:
        raise HTTPException(409, "The review is still running.")
    reviewer = str(payload.get("reviewer", "")).strip()
    decision = str(payload.get("decision", "")).strip()
    if not reviewer:
        raise HTTPException(400, "A named reviewer is required. The agent cannot approve.")
    if decision not in ("approved", "changes_requested"):
        raise HTTPException(400, "Decision must be approved or changes_requested.")
    record = {"status": decision, "reviewer": reviewer, "note": str(payload.get("note", "")).strip()[:1000],
              "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "open_failures": sum(1 for c in job.result["claims"] if c["status"] == "fail")}
    job.result["approval"] = record
    return record
