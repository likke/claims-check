# Dasho Claim Check

A pre-publication review prototype for creator videos. Brand teams upload a 15–90 second video, add product sources and key messages, then a named reviewer approves it or requests changes. Built for marketing and compliance teams running Tech claims or FDA beauty checks.

## Run

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. With `MODEL_API_KEY` unset, the app uses an offline fictional Ledgerly AI fixture. Set `MODEL_API_KEY`, `MODEL_BASE_URL`, and `MODEL_NAME` to select an OpenAI-compatible Muse Spark Open endpoint. Do not upload confidential media to a hosted endpoint without approval.

## Architecture

FastAPI exposes the upload/review and local decision endpoints; `static/index.html` is the single page. The provider interface supplies speech, sampled-frame observations, and agent planning. A bounded plan/act/observe runner logs tool actions and returns source evidence, disclosure, and coverage checks. The fixture provider supports offline demos. `ffmpeg` is an expected media extraction integration point; current fixture mode does not decode uploaded media.

## Meta models

The intended cloud integration uses Muse Voice Transcribe for Taglish speech, Muse Spark for the traceable agent, and Muse Spark vision for three to six sampled frames. The OpenAI-compatible provider switch is provided for local/open-weight deployments. This fresh repo currently runs the deterministic fixture unless configured; live Meta API calls require endpoint-specific credentials and request wiring.

## Built during the hackathon

- 26 Oct 2026 — Created the fresh app structure, offline provider, review workflow, and tests.

A same-day prototype (26 Oct, before kickoff) existed in a separate private repo; no code from it is used here.

## License

AGPL-3.0; see [LICENSE](LICENSE).
