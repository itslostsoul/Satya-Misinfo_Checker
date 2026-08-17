import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from orchestrator import classify_and_route
from verdict import merge_and_calibrate
from card_renderer import render_card
from mock_responses import mock_verify

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_valid_key(key: str) -> bool:
    return bool(key) and not key.strip().lower().startswith("your_")

HAS_KEYS = is_valid_key(os.getenv("ANTHROPIC_API_KEY")) and is_valid_key(os.getenv("GOOGLE_FACT_CHECK_KEY"))

@app.post("/api/verify")
async def verify(
    text: str = Form(None),
    image: UploadFile = File(None),
    language: str = Form("tamil")
):
    if not HAS_KEYS:
        print("[WARN] API keys missing — running in mock mode")
        return await mock_verify(text, image)

    image_bytes = await image.read() if image else None
    pipeline_results = await classify_and_route(text=text, image_bytes=image_bytes)
    verdict = merge_and_calibrate(pipeline_results)
    card = await render_card(verdict, language)
    return card

app.mount("/", StaticFiles(directory="../public", html=True), name="static")
