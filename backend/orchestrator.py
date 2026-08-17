import asyncio
import logging
import os
from pipelines.image_pipeline import run_image_pipeline
from pipelines.text_pipeline import run_text_pipeline
from pipelines.claim_extractor import extract_claim
from pipelines.fact_checker import search_fact_checks
from pipelines.verdict_engine import synthesize_verdict

logger = logging.getLogger(__name__)

async def classify_and_route(text: str, image_bytes: bytes) -> dict:
    tasks = []
    keys = []

    if image_bytes:
        tasks.append(run_image_pipeline(image_bytes))
        keys.append("image_result")
    if text:
        # Use Claude + Scraping pipeline if ANTHROPIC_API_KEY is available,
        # otherwise fallback to Gemini + Google Fact Check tools API
        if os.getenv("ANTHROPIC_API_KEY"):
            tasks.append(analyze_text(text))
        else:
            tasks.append(run_text_pipeline(text))
        keys.append("text_result")

    results = await asyncio.gather(*tasks)

    ret = {}
    for i in range(len(keys)):
        val = results[i]
        # Calibrate confidence if it is expressed as a float <= 1.0
        if keys[i] == "text_result" and val:
            if "confidence" in val and val["confidence"] is not None:
                if val["confidence"] <= 1.0:
                    val["confidence"] = int(val["confidence"] * 100)
        ret[keys[i]] = val

    return ret

async def analyze_text(text: str) -> dict:
    """
    Executes the end-to-end text claim analysis pipeline:
    claim extraction -> fact checking -> verdict synthesis.
    
    Args:
        text (str): The raw text of the message.
        
    Returns:
        dict: The final verdict details.
    """
    # Stage 1: Claim extraction
    try:
        extraction_result = await extract_claim(text)
    except Exception as e:
        logger.exception("Claim extraction failed in orchestrator")
        return {
            "verdict": "unverifiable",
            "confidence": 0.0,
            "reason": f"Failed to extract claim from text: {str(e)}",
            "sources": [],
            "claim": None,
            "calibration_note": f"Exception in claim_extractor: {str(e)}"
        }
        
    claim = extraction_result.get("claim")
    
    # If no claim extracted (null), short circuit
    if not claim:
        return {
            "verdict": "unverifiable",
            "confidence": 0.0,
            "reason": "No specific factual claim found",
            "sources": [],
            "claim": None,
            "calibration_note": "Short-circuited because claim is null."
        }
        
    # Stage 2: Fact checking
    try:
        search_result = await search_fact_checks(claim)
        fact_checks = search_result.get("results", [])
    except Exception as e:
        logger.exception("Fact checking failed in orchestrator")
        # Proceed with empty fact checks list so verdict engine handles calibration
        fact_checks = []
        
    # Stage 3: Verdict synthesis
    try:
        verdict_result = await synthesize_verdict(text, claim, fact_checks)
        return verdict_result
    except Exception as e:
        logger.exception("Verdict synthesis failed in orchestrator")
        return {
            "verdict": "unverifiable",
            "confidence": 0.0,
            "reason": "Failed to synthesize final verdict",
            "sources": [],
            "claim": claim,
            "calibration_note": f"Exception in verdict_engine: {str(e)}"
        }
