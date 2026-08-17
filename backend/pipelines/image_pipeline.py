import io
import os
import httpx
from datetime import date, datetime
from PIL import Image
from transformers import pipeline as hf_pipeline

_deepfake_detector = None


def get_deepfake_detector():
    # Loaded lazily (not at import time) so the server can boot — and mock
    # mode can serve requests — without pulling ~300MB from HF Hub first.
    global _deepfake_detector
    if _deepfake_detector is None:
        _deepfake_detector = hf_pipeline(
            "image-classification",
            model="umm-maybe/AI-image-detector"
        )
    return _deepfake_detector

async def run_image_pipeline(image_bytes: bytes) -> dict:
    # Step 1: AI/deepfake detection
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ai_result = get_deepfake_detector()(image)
    is_ai = ai_result[0]['label'].lower() in ['artificial', 'fake', 'ai-generated']
    ai_confidence = float(ai_result[0]['score'])

    # Step 2: Reverse image search
    serp_result = await reverse_image_search(image_bytes)
    earliest_date = serp_result.get("earliest_date")

    # Step 3: Date mismatch
    date_mismatch = False
    delta_days = 0
    if earliest_date:
        try:
            parsed = datetime.strptime(earliest_date, "%Y-%m-%d").date()
            delta_days = (date.today() - parsed).days
            date_mismatch = delta_days > 30
        except Exception:
            pass

    return {
        "is_ai_generated": is_ai,
        "ai_confidence": ai_confidence,
        "date_mismatch": date_mismatch,
        "delta_days": delta_days,
        "original_context": serp_result.get("context", ""),
        "source_urls": serp_result.get("source_urls", [])
    }

async def upload_temp_image(image_bytes: bytes) -> str:
    # SerpAPI's reverse-image engines take a public image URL, not raw bytes,
    # so the image has to be hosted somewhere public first.
    #
    # KNOWN BLIND SPOT: 0x0.st has permanently disabled anonymous uploads
    # ("uploads disabled ... no ETA", checked 2026-08-17). catbox.moe and
    # tmpfiles.org were also tried and are dead/blocked. Until someone signs
    # up for a free imgbb.com key (or similar) and wires it in here, this
    # call will fail and reverse_image_search() below degrades to
    # "unverifiable" — which is why it's wrapped in try/except, not a bug.
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://0x0.st",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")}
        )
    resp.raise_for_status()
    return resp.text.strip()

async def reverse_image_search(image_bytes: bytes) -> dict:
    SERP_KEY = os.getenv("SERPAPI_API_KEY")
    if not SERP_KEY:
        return {}

    try:
        image_url = await upload_temp_image(image_bytes)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={"engine": "google_reverse_image", "image_url": image_url, "api_key": SERP_KEY}
            )
        return parse_serp_response(resp.json())
    except Exception as e:
        # Expected right now — see upload_temp_image()'s docstring. This is a
        # documented blind spot, not a silent failure: the pipeline still
        # returns a valid (unverifiable) result instead of crashing.
        print(f"[WARN] Reverse image search unavailable (image host down): {e}")
        return {}

def parse_serp_response(data: dict) -> dict:
    results = data.get("image_results", [])
    source_urls = [r.get("link") for r in results[:3] if r.get("link")]
    earliest_date = None
    context = ""

    for r in results:
        if r.get("date"):
            earliest_date = r["date"]
            context = r.get("title", "")
            break

    return {
        "earliest_date": earliest_date,
        "context": context,
        "source_urls": source_urls
    }
