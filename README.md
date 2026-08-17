# Satya — AI Forward-Checker (PS-S03)

A FastAPI backend + share-target PWA frontend that checks a forwarded
text claim or image and returns a plain-language verdict card, in
English and a regional language, with sources.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # then fill in your API keys
cd backend
uvicorn server:app --reload
```

Open http://127.0.0.1:8000 — it's the same FastAPI app serving the PWA
frontend from `public/`.

**No API keys? No problem.** With `GEMINI_API_KEY` and
`GOOGLE_FACT_CHECK_KEY` unset, the server automatically serves mock
verdicts (see `backend/mock_responses.py`) so the full UI flow works
without any network calls.

## Architecture

| File | Job |
|---|---|
| [backend/server.py](backend/server.py) | FastAPI app, `/api/verify` endpoint, mock-mode fallback |
| [backend/orchestrator.py](backend/orchestrator.py) | Fans out to image/text pipelines concurrently |
| [backend/pipelines/image_pipeline.py](backend/pipelines/image_pipeline.py) | AI/deepfake detection + reverse image search (SerpAPI) |
| [backend/pipelines/text_pipeline.py](backend/pipelines/text_pipeline.py) | Claim extraction (Gemini) + fact-check source matching |
| [backend/pipelines/voice_pipeline.py](backend/pipelines/voice_pipeline.py) | Stretch goal, stub only |
| [backend/verdict.py](backend/verdict.py) | Merges pipeline signals into one calibrated verdict |
| [backend/card_renderer.py](backend/card_renderer.py) | Writes the final two-line English + regional-language card (Gemini) |
| [public/](public/) | The PWA frontend `server.py` serves |

Heavy ML models (deepfake detector, sentence embedder) load lazily on
first real request, not at import — so the server boots instantly and
mock mode never touches the network or disk cache.

## Environment variables (see `.env.example`)

- `GEMINI_API_KEY` — claim extraction + card writing (free tier: https://aistudio.google.com)
- `SERPAPI_API_KEY` — reverse image search (https://serpapi.com, free tier: 100/mo)
- `GOOGLE_FACT_CHECK_KEY` — fact-check source lookup (Google Fact Check Tools API)

Any subset can be missing — each pipeline degrades to `unverifiable`
for what it can't check rather than crashing.

**Never commit `.env`.** It's gitignored — if you rotate a leaked key,
generate a new one and only put it in your local `.env`.

## Known blind spot: reverse image search

`SERPAPI_API_KEY` alone isn't enough — SerpAPI's reverse-image engines
require a **public image URL**, not raw file bytes, so the app has to
host the image somewhere first. As of 2026-08-17, the three free
anonymous hosts tried (`0x0.st`, `catbox.moe`, `tmpfiles.org`) are all
dead or blocking uploads. Until someone wires in a real host (e.g. a
free `imgbb.com` API key), `image_pipeline.py`'s reverse-context check
silently degrades to `unverifiable` — see the comment in
`upload_temp_image()`. AI-generation detection (the other half of the
image pipeline) is unaffected and works standalone.

## Before the demo

- Run against the 8-item judging set repeatedly, not just once.
- Confirm real replies (not mock) land under 60s once keys are set.
- `blind_spots` / degraded-pipeline messaging is part of the judged deliverables — don't hide gaps (see above: this one's already documented for you).
