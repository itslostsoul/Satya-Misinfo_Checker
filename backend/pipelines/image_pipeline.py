import os
import httpx
from datetime import date, datetime
from transformers import pipeline as hf_pipeline

deepfake_detector = hf_pipeline(
    "image-classification",
    model="umm-maybe/AI-image-detector"
)

async def run_image_pipeline(image_bytes: bytes) -> dict:
    # Step 1: AI/deepfake detection
    try:
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_bytes))
        ai_result = deepfake_detector(image)
        is_ai = ai_result[0]['label'].lower() in ['artificial', 'fake', 'ai-generated']
        ai_confidence = float(ai_result[0]['score'])
    except Exception as e:
        print(f"[WARN] Deepfake detection failed to parse image: {e}")
        is_ai = False
        ai_confidence = 0.0

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

async def reverse_image_search(image_bytes: bytes) -> dict:
    SERP_KEY = os.getenv("SERPAPI_KEY")
    if not SERP_KEY:
        return {}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://serpapi.com/search",
                params={"engine": "google_reverse_image", "api_key": SERP_KEY},
                files={"image": ("image.jpg", image_bytes, "image/jpeg")}
            )
        return parse_serp_response(resp.json())
    except Exception as e:
        print(f"[WARN] Reverse image search failed: {e}")
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
