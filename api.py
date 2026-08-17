"""
Image Forensics REST API (FastAPI).

Provides high-performance image authenticity & manipulation analysis via HTTP:
- POST /analyze : Analyze image via file upload or URL
- GET /health   : Service health check
- GET /info     : Detector capabilities & device information
"""

import io
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import httpx
from PIL import Image

from forensics.fusion import analyze_image_forensics
from forensics.ai_detector import ai_detector


app = FastAPI(
    title="Image Forensics Engineer API",
    description="Multi-layer Image Authenticity, AI-Generation, and Tampering Detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeURLRequest(BaseModel):
    image_url: HttpUrl
    claimed_source_url: Optional[str] = None
    is_screenshot: bool = False


async def load_image_from_bytes(data: bytes) -> Image.Image:
    """Safely opens image bytes into a PIL Image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image data: {e}"
        )


async def fetch_image_from_url(url: str) -> Image.Image:
    """Downloads image from a remote URL with strict timeout and validation."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return await load_image_from_bytes(resp.content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch image from URL '{url}': {e}"
        )


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "image-forensics-api",
        "version": "1.0.0",
    }


@app.get("/info", tags=["Capabilities"])
async def service_info():
    """Returns active forensic modules, model states, and hardware acceleration."""
    return {
        "service": "Image Forensics Engineer",
        "detectors": [
            {
                "name": "Error Level Analysis (ELA)",
                "type": "Compression Forensics & Spatial Block Variance",
                "status": "active"
            },
            {
                "name": "EXIF / IPTC / XMP Provenance Scanner",
                "type": "Software Tags & GenAI Signature Forensics",
                "status": "active"
            },
            {
                "name": "AI-Generation Classifier",
                "type": "Deep Learning / 2D FFT Spectral Artifacts",
                "model_name": ai_detector.model_name,
                "status": ai_detector.detect(Image.new("RGB", (16, 16))).get("status")
            },
            {
                "name": "Face & Deepfake Detector",
                "type": "Facial Boundary Seam & Texture Blur Analysis",
                "status": "active"
            },
            {
                "name": "Doctored Screenshot & Chyron Detector",
                "type": "Aspect Ratio & Edge Sharpness Tampering Analysis",
                "status": "active"
            }
        ]
    }


@app.post("/analyze", tags=["Forensics Analysis"])
async def analyze_image(
    image_file: Optional[UploadFile] = File(None, description="Image file to analyze"),
    image_url: Optional[str] = Form(None, description="Direct URL of image to analyze"),
    claimed_source_url: Optional[str] = Form(None, description="Optional publisher URL to cross-check headline against"),
    is_screenshot: bool = Form(False, description="Flag if input is explicitly known to be a screenshot")
):
    """
    Main image forensics analysis endpoint.
    Accepts either multipart file upload or image URL.
    """
    if not image_file and not image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'image_file' upload or 'image_url' must be provided."
        )

    if image_file:
        file_bytes = await image_file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )
        img = await load_image_from_bytes(file_bytes)
    else:
        img = await fetch_image_from_url(image_url)  # type: ignore[arg-type]

    # Run complete forensic analysis pipeline
    result = analyze_image_forensics(
        image=img,
        claimed_source_url=claimed_source_url,
        force_screenshot=is_screenshot
    )

    return result


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with quicklinks to Swagger UI and health."""
    return {
        "message": "Image Forensics Engineer API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
        "info_url": "/info",
        "analyze_url": "/analyze"
    }


@app.post("/analyze/json", tags=["Forensics Analysis"])
async def analyze_image_json(req: AnalyzeURLRequest):
    """JSON payload endpoint for URL-based analysis."""
    img = await fetch_image_from_url(str(req.image_url))
    return analyze_image_forensics(
        image=img,
        claimed_source_url=req.claimed_source_url,
        force_screenshot=req.is_screenshot
    )


@app.post("/api/verify", tags=["Satya Integration"])
async def verify_forward(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    language: str = Form("english")
):
    """
    Drop-in compatibility endpoint for the Satya Misinfo-Checker Web UI.
    """
    if not image and not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either text or image must be provided."
        )

    if image:
        file_bytes = await image.read()
        img = await load_image_from_bytes(file_bytes)
        result = analyze_image_forensics(img)
        verdict_str = "false" if result["verdict"] == "manipulated" else ("true" if result["verdict"] == "authentic" else "unverifiable")
        conf_int = int(result["confidence"] * 100) if result["verdict"] != "uncertain" else None

        return {
            "verdict": verdict_str,
            "confidence": conf_int,
            "explanation_en": result["reason"],
            "explanation_regional": result["reason"],
            "sources": [
                {"title": "Image Forensics (ELA & AI Detector)", "url": "#"}
            ],
            "signals": result["signals"]
        }
    else:
        return {
            "verdict": "unverifiable",
            "confidence": None,
            "explanation_en": f"Text received: '{text[:80]}...'. For deep text fact-checks, configure Google Fact Check API keys.",
            "explanation_regional": "Text claim verification requires fact-check database connectivity.",
            "sources": []
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
