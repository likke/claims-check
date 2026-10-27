"""FastAPI app: start a review, stream its trace over SSE, record a human decision."""
from __future__ import annotations

import asyncio
import json
import tempfile
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import fixture
from .agent import run_review
from .media import FORMATS, MAX_CHECK_SECONDS, MediaError, extract_media, image_frame, make_proxy, needs_proxy, probe, timeline_art
from .packs import get_pack
from . import demo
from .providers import get_provider

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 80 * 1024 * 1024

app = FastAPI(title="Dasho Claim Check")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


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
        self.video_path: Path | None = None
        self.folder: Path | None = None
        self.original: Path | None = None

    def push(self, event: dict):
        with self.lock:
            self.events.append(event)


JOBS: dict[str, Job] = {}


def _worker(job: Job, provider, video: bytes | None, suffix: str, params: dict):
    try:
        wav, frames, extra = b"", [], {}
        if not provider.needs_media:
            frames = provider.frames()
        if video:
            folder = Path(tempfile.mkdtemp(prefix=f"claimcheck-{job.id}-"))
            original = folder / f"original{suffix}"
            original.write_bytes(video)
            job.folder, job.original = folder, original
            info = probe(original)
            proxy = make_proxy(original, folder / "proxy.mp4", info) if needs_proxy(info) else None
            job.video_path = proxy or original
            job.push({"seq": 0, "t": 0, "kind": "media_info", **info, "proxy": bool(proxy), "layout": f"layout-{info['orientation']}",
                      "video_url": f"/api/reviews/{job.id}/video",
                      "summary": f"{info['display_width']}x{info['display_height']} {info['orientation']}, {info['video_codec']}"
                                 + (f", rotated {info['rotation']} degrees" if info["rotation"] else "")
                                 + (", playing a browser-safe H.264 copy" if proxy else "")})
            job.push({"seq": 0, "t": 0, "kind": "art", **timeline_art(job.video_path, min(info["duration"] or 1, MAX_CHECK_SECONDS))})
            extra["media"] = info
            if provider.needs_media:
                media = extract_media(video, info)
                if media["trimmed"]:
                    job.push({"seq": 0, "t": 0, "kind": "warning",
                              "summary": f"The clip is {media['full_duration']:.0f} s long; only the first {MAX_CHECK_SECONDS} s are checked."})
                if not info["has_audio"]:
                    extra.update(no_audio=True, duration=media["duration"])
                wav, frames = media["wav"], media["frames"]
                job.push({"seq": 0, "t": 0, "kind": "frames", "frames": [{"time": f["time"], "image": f["image"]} for f in frames]})
        elif provider.needs_media:
            raise MediaError("Upload a video (15 to 90 seconds).")
        job.result = run_review(provider, wav=wav, frames=frames, emit=job.push, **params, **extra)
    except Exception as error:
        job.error = f"{type(error).__name__}: {error}"
    finally:
        job.done = True
        demo.cleanup(job)


@app.get("/", response_class=HTMLResponse)
@app.get("/review", response_class=HTMLResponse)
def home():
    """The review screen (default)."""
    return (ROOT / "static" / "review.html").read_text(encoding="utf-8")


@app.get("/classic", response_class=HTMLResponse)
def classic():
    """The earlier agent-app screen, kept until the review screen replaces it."""
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config(request: Request):
    provider = get_provider()
    return {
        "provider": provider.info(),
        "needs_video": provider.needs_media,
        "packs": ["Tech claims", "FDA beauty"],
        "example": {"product": fixture.PRODUCT, "sources": fixture.SOURCE_PACK, "key_messages": "\n".join(fixture.KEY_MESSAGES)},
        "demo": demo.public_limits(request),
    }


def replay_config() -> dict:
    """Which recorded run and clip the replay uses.

    Read from demo/replay.json ({"run": "...", "clip": "..."}, paths relative
    to the repo), overridable with REPLAY_RUN and REPLAY_CLIP. To demo a new
    clip: run demo/live_check.py --video new.mp4 --update-replay. No code change.
    """
    config = {}
    path = Path(os.getenv("REPLAY_CONFIG", ROOT / "demo" / "replay.json"))
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
    run = os.getenv("REPLAY_RUN") or config.get("run")
    clip = os.getenv("REPLAY_CLIP") or config.get("clip")
    if not run:
        runs = sorted((ROOT / "demo").glob("live-run-*.json"))
        run = str(runs[-1]) if runs else None
    resolve = lambda value: (Path(value) if Path(value).is_absolute() else ROOT / value) if value else None
    return {"run": resolve(run), "clip": resolve(clip), "label_box": config.get("label_box")}


@app.api_route("/api/replay", methods=["GET", "HEAD"])
def replay():
    """The configured recorded run, for filming without spending API credits."""
    paths = replay_config()
    if not paths["run"] or not paths["run"].exists():
        raise HTTPException(404, "No recorded live run")
    run = json.loads(paths["run"].read_text(encoding="utf-8"))
    if "inputs" not in run:  # runs recorded before inputs were saved: rebuild them from the run itself
        run["inputs"] = {
            "product": run.get("product", ""), "pack": run.get("pack", "Tech claims"), "supplement": run.get("supplement", False),
            "sources": " ".join(c["text"] for c in run.get("sources", []) if c.get("origin") == "pasted")
                       + "".join("\n" + url for url in run.get("fetched_urls", {})),
            "key_messages": [m["message"] for m in (run.get("coverage") or {}).get("messages", [])],
        }
    clip = paths["clip"]
    has_clip = bool(clip and clip.exists())
    if has_clip:
        info = probe(clip)
        playable = _replay_playable(clip, info)
        run["media"] = {**info, "proxy": playable != clip, "layout": f"layout-{info['orientation']}"}
    run["video_url"] = "/api/replay/video" if has_clip else None
    run["video_name"] = clip.name if has_clip else None
    # Where the paid-partnership label sits in this clip, when the config knows it (our generated test clip does).
    run["label_box"] = paths["label_box"] if has_clip else None
    run["frame_images"] = _frame_images(clip, [f["time"] for f in run.get("frames", [])]) if has_clip else []
    run["art"] = timeline_art(clip, run.get("duration") or 1.0) if has_clip else {}
    return run


def _frame_images(clip: Path, times: list[float]) -> list[dict]:
    """Thumbnails for the recorded run, cut from the clip at the sampled frame times."""
    import base64
    import subprocess
    images = []
    for at in times:
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{at:.2f}", "-i", str(clip), "-frames:v", "1", "-vf", "scale=480:-2",
                              "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"], capture_output=True)
        if out.returncode == 0 and out.stdout:
            images.append({"time": at, "image": base64.b64encode(out.stdout).decode("ascii")})
    return images


def _replay_playable(clip: Path, info: dict) -> Path:
    """The replay clip, or a cached H.264 copy of it when the browser may not play the original."""
    if not needs_proxy(info):
        return clip
    stat = clip.stat()
    cached = ROOT / ".cache" / f"{clip.stem}-{int(stat.st_mtime)}-{stat.st_size}.mp4"
    if not cached.exists():
        cached.parent.mkdir(exist_ok=True)
        make_proxy(clip, cached, info)
    return cached


@app.get("/api/replay/video")
def replay_video():
    clip = replay_config()["clip"]
    if not clip or not clip.exists():
        raise HTTPException(404, "No recorded clip")
    clip = _replay_playable(clip, probe(clip))
    media = {".mov": "video/quicktime", ".webm": "video/webm"}.get(clip.suffix.lower(), "video/mp4")
    return FileResponse(clip, media_type=media)


IMAGE_FORMATS = {".jpg", ".jpeg", ".png"}
MAX_IMAGES = 10
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@app.post("/api/reviews")
async def start_review(
    request: Request,
    video: UploadFile | None = File(None),
    images: list[UploadFile] | None = File(None),
    sources: str = Form(""),
    key_messages: str = Form(""),
    pack: str = Form("Tech claims"),
    product: str = Form(""),
    supplement: str = Form(""),
    caption: str = Form(""),
):
    provider = get_provider()
    data = await video.read() if video is not None else b""
    suffix = Path(video.filename or "").suffix.lower() if video is not None else ""
    pictures = []
    for upload in images or []:
        content = await upload.read()
        if not content:
            continue
        if Path(upload.filename or "").suffix.lower() not in IMAGE_FORMATS:
            raise HTTPException(415, "Images must be JPG or PNG.")
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Each image must be under 10 MB.")
        pictures.append(content)
    if data and pictures:
        raise HTTPException(400, "Upload a video or images, not both.")
    if len(pictures) > MAX_IMAGES:
        raise HTTPException(400, f"Upload at most {MAX_IMAGES} images.")
    if data and suffix not in FORMATS:
        raise HTTPException(415, "Upload an MP4, MOV, M4V or WebM video.")
    if provider.needs_media and not data and not pictures:
        raise HTTPException(400, "Upload a video (15 to 90 seconds) or the post images.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "The video is larger than 80 MB.")
    if not sources.strip():
        raise HTTPException(400, "Add a source pack: product docs or URLs.")
    if len(caption) > 4000:
        raise HTTPException(400, "Keep the caption under 4,000 characters.")
    demo.check_upload(request, data, suffix, sum(len(p) for p in pictures))
    job = Job(provider.info())
    JOBS[job.id] = job
    params = {
        "sources": sources, "key_messages": key_messages.splitlines(), "pack": get_pack(pack),
        "product": product.strip(), "supplement": supplement.lower() in ("1", "true", "on", "yes"),
        "caption": caption.strip(),
    }
    if pictures:
        threading.Thread(target=_image_worker, args=(job, provider, pictures, params), daemon=True).start()
    else:
        threading.Thread(target=_worker, args=(job, provider, data, suffix, params), daemon=True).start()
    return {"id": job.id, "provider": provider.info()}


def _image_worker(job: Job, provider, pictures: list[bytes], params: dict):
    """An image or carousel post: each image is a numbered frame (image 1, 2, 3); there is no timeline."""
    folder = Path(tempfile.mkdtemp(prefix=f"claimcheck-{job.id}-"))
    try:
        frames = []
        for number, content in enumerate(pictures, start=1):
            frames.append(image_frame(content, number, folder))
        job.folder = folder
        job.push({"seq": 0, "t": 0, "kind": "media_info", "orientation": "images", "layout": "layout-images", "images": len(frames),
                  "summary": f"Image post with {len(frames)} image{'s' if len(frames) != 1 else ''}"})
        job.push({"seq": 0, "t": 0, "kind": "post_images", "images": [{"number": f["time"], "image": f["image"], "width": f["width"],
                                                                       "height": f["height"]} for f in frames]})
        job.result = run_review(provider, frames=frames, emit=job.push, post_kind="images",
                                media={"orientation": "images", "images": len(frames)}, **params)
    except Exception as error:
        job.error = f"{type(error).__name__}: {error}"
    finally:
        job.done = True
        demo.cleanup(job)


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


@app.get("/api/reviews/{review_id}/video")
def review_video(review_id: str):
    """The clip as the browser should play it: the H.264 copy when one was made, else the original."""
    job = _job(review_id)
    if not job.video_path or not job.video_path.exists():
        raise HTTPException(404, "No video for this review")
    media = {".mov": "video/quicktime", ".webm": "video/webm", ".m4v": "video/mp4"}.get(job.video_path.suffix.lower(), "video/mp4")
    return FileResponse(job.video_path, media_type=media)


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


@app.post("/api/ask")
def ask(payload: dict, request: Request):
    """Answer a question about a finished review from its result JSON, with the configured model."""
    job = _job(str(payload.get("review_id", "")))
    demo.check_ask(request)
    if not job.done or job.result is None:
        raise HTTPException(409, "The review is still running.")
    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(400, "Ask a question.")
    if len(question) > 500:
        raise HTTPException(400, "Keep the question under 500 characters.")
    from . import prompts
    provider = get_provider()
    try:
        data = provider.complete_json("ask", prompts.ask(question, job.result), context={"question": question, "run": job.result})
    except Exception as error:
        raise HTTPException(502, f"The model could not answer: {error}")
    return {"answer": str(data.get("answer", "")).strip() or "No answer.", "provider": provider.info()["name"]}


def _art_from_bytes(video: bytes, duration: float) -> dict:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="claimcheck-art-") as tmp:
        path = Path(tmp) / "clip"
        path.write_bytes(video)
        return timeline_art(path, duration)
