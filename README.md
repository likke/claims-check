# Dasho Claim Check

A pre-publication review prototype for creator videos. Brand teams upload a 15–90 second video, add product sources and key messages, then a named reviewer approves it or requests changes. Built for marketing and compliance teams running Tech claims or FDA beauty checks.

## Run

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. With `MODEL_API_KEY` unset, the app uses an offline fictional Ledgerly AI fixture. Set `MODEL_API_KEY` to use the Meta API (`https://api.meta.ai/v1`); optional `MODEL_BASE_URL`, `MODEL_NAME`, and `ASR_MODEL` override endpoint/model selection. Video uploads require `ffmpeg` and `ffprobe` on PATH.

## Architecture

FastAPI exposes upload/review, trace polling, and SSE trace endpoints; `static/index.html` is the single page. Live media extraction produces mono 16 kHz audio and up to six evenly spaced frames, then cleans its temporary directory. The bounded agent runner logs tool actions and returns source evidence, disclosure, and coverage checks. The fixture provider supports offline demos.

## Meta models

Live mode uses Muse Voice Transcribe for Tagalog/English biased transcription, Muse Spark for plan/act/observe and claim decisions, rewrites, disclosure and coverage, and Muse Spark vision for sampled frames. It tries native function tools and falls back to JSON planning if the endpoint rejects tools. The offline fixture remains available when `MODEL_API_KEY` is unset.

## Built during the hackathon

- 26 Oct 2026 — Created the fresh app structure, offline provider, review workflow, and tests.
- 27 Oct 2026 — Wired Meta Muse Voice Transcribe and Muse Spark, ffmpeg media extraction, trace events, and a fictional labeled eval harness.

A same-day prototype (26 Oct, before kickoff) existed in a separate private repo; no code from it is used here.

## License

AGPL-3.0; see [LICENSE](LICENSE).
