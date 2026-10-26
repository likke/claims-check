"""FastAPI application and review routes."""
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import run_review
from .packs import TECH_SOURCE

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Dasho Claim Check")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
REVIEWS = {}


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text()


@app.post("/api/review")
async def review(video: UploadFile = File(...), sources: str = Form(TECH_SOURCE), key_messages: str = Form("Organize invoices\nFlag unusual charges"), pack: str = Form("Tech claims")):
    result = run_review(await video.read(), sources, key_messages, pack)
    REVIEWS[result["id"]] = result
    return result


@app.get("/api/review/{review_id}/trace")
def trace(review_id):
    result = REVIEWS.get(review_id)
    if not result:
        return {"error": "Review not found"}
    return {"trace": result["trace"]}


@app.get("/api/review/{review_id}/events")
async def events(review_id):
    async def stream():
        for item in REVIEWS.get(review_id, {}).get("trace", []):
            yield "data: " + json.dumps(item) + "\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/review/{review_id}/decision")
async def decision(review_id, payload: dict):
    if review_id not in REVIEWS:
        return {"error": "Review not found"}
    REVIEWS[review_id]["decision"] = {"status": payload.get("status"), "reviewer": payload.get("reviewer", "")}
    return REVIEWS[review_id]["decision"]
