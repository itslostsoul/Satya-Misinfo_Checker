import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from orchestrator import classify_and_route
from verdict import merge_and_calibrate
from card_renderer import render_card
from mock_responses import mock_verify

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HAS_KEYS = (bool(os.getenv("GEMINI_API_KEY")) and bool(os.getenv("GOOGLE_FACT_CHECK_KEY"))) or bool(os.getenv("ANTHROPIC_API_KEY"))

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

from pydantic import BaseModel
from fastapi import HTTPException

class AnalyzeTextRequest(BaseModel):
    text: str

@app.post("/analyze-text")
async def analyze_text_endpoint(req: AnalyzeTextRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.text) > 5000:
        raise HTTPException(status_code=400, detail="Text exceeds 5000 characters limit")
        
    from orchestrator import analyze_text
    try:
        result = await analyze_text(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="../public", html=True), name="static")
