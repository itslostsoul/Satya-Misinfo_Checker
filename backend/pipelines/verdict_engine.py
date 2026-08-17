import json
import logging
import os
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

# Load env variables
load_dotenv()

logger = logging.getLogger(__name__)

# Calibration Constants
CONFIDENCE_THRESHOLD = 0.6
DEFAULT_VERDICT_NULL_CLAIM = "unverifiable"
DEFAULT_REASON_NULL_CLAIM = "No specific factual claim found"
DEFAULT_VERDICT_ZERO_RESULTS = "unverifiable"
DEFAULT_REASON_ZERO_RESULTS = "No matching fact-checks found"
MODEL_NAME = "claude-sonnet-4-6"

async def synthesize_verdict(original_text: str, claim: str | None, fact_checks: list[dict]) -> dict:
    """
    Synthesize claim + fact-check results into a final calibrated verdict using Anthropic Claude.
    
    Args:
        original_text (str): The raw original text forward.
        claim (str or None): The extracted factual claim.
        fact_checks (list[dict]): The list of search result articles.
        
    Returns:
        dict: The calibrated verdict dictionary containing verdict, confidence, reason,
              sources, claim, and calibration_note.
    """
    # 1. Check if claim is null
    if not claim:
        return {
            "verdict": DEFAULT_VERDICT_NULL_CLAIM,
            "confidence": 0.0,
            "reason": DEFAULT_REASON_NULL_CLAIM,
            "sources": [],
            "claim": None,
            "calibration_note": "Claim is null; short-circuited to unverifiable."
        }
        
    # 2. Check if 0 fact-check results found
    if not fact_checks:
        return {
            "verdict": DEFAULT_VERDICT_ZERO_RESULTS,
            "confidence": 0.0,
            "reason": DEFAULT_REASON_ZERO_RESULTS,
            "sources": [],
            "claim": claim,
            "calibration_note": "No search results returned; short-circuited to unverifiable."
        }
        
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment")
        return {
            "verdict": "unverifiable",
            "confidence": 0.0,
            "reason": "API key missing for verdict synthesis",
            "sources": [],
            "claim": claim,
            "calibration_note": "No ANTHROPIC_API_KEY configured."
        }
        
    client = AsyncAnthropic(api_key=api_key)
    
    system_prompt = (
        "You are an expert fact-check synthesis system. You analyze an extracted claim, the original WhatsApp forward text, "
        "and a list of search results from fact-checking websites, and output a structured verdict.\n\n"
        "Rules:\n"
        "1. Determine whether the search results confirm, deny, or are unrelated to the claim.\n"
        "2. Choose one of the valid verdicts:\n"
        "   - \"true\": The search results confirm the claim is correct.\n"
        "   - \"false\": The search results debunk the claim as false or fake.\n"
        "   - \"misleading\": The claim is partly true but presented in a deceptive or out-of-context way.\n"
        "   - \"unverifiable\": The search results do not provide enough direct evidence to confirm or deny the claim.\n"
        "3. Provide a confidence score between 0.0 and 1.0 based on the relevance and strong matching of the fact check evidence to the claim.\n"
        "4. Provide a 2-3 sentence explanation in plain English summarizing the reasoning.\n"
        "5. Include a list of supporting sources from the input search results. Format each source with \"title\", \"url\", and \"source\".\n"
        "6. Output must be a valid JSON object only. Do not wrap in markdown backticks, no preamble, and no explanation. Output ONLY the JSON string.\n"
        "7. The JSON keys must be exactly: \"verdict\", \"confidence\", \"reason\", \"sources\", \"calibration_note\"."
    )
    
    user_content = (
        f"Original Forward Text:\n{original_text}\n\n"
        f"Extracted Claim:\n{claim}\n\n"
        f"Fact-Check Search Results:\n{json.dumps(fact_checks, indent=2)}\n"
    )
    
    try:
        response = await client.messages.create(
            model=MODEL_NAME,
            max_tokens=600,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )
        raw_output = response.content[0].text.strip()
        
        # Clean up any potential markdown formatting added by the model
        if raw_output.startswith("```"):
            lines = raw_output.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_output = "\n".join(lines).strip()
            
        parsed = json.loads(raw_output)
        
        verdict = parsed.get("verdict", "unverifiable")
        confidence = float(parsed.get("confidence", 0.0))
        reason = parsed.get("reason", "")
        sources = parsed.get("sources", [])
        calibration_note = parsed.get("calibration_note", "")
        
        # Calibration rule: If Claude confidence < 0.6 -> override verdict to "unverifiable"
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Confidence score {confidence} is below threshold {CONFIDENCE_THRESHOLD}. Overriding verdict to unverifiable.")
            calibration_note = (
                f"Confidence score {confidence} is below threshold {CONFIDENCE_THRESHOLD}. "
                f"Verdict overridden from '{verdict}' to 'unverifiable'. "
                f"Original note: {calibration_note}"
            )
            verdict = "unverifiable"
            
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "sources": sources,
            "claim": claim,
            "calibration_note": calibration_note
        }
        
    except Exception as e:
        logger.exception("Error in verdict engine synthesis")
        return {
            "verdict": "unverifiable",
            "confidence": 0.0,
            "reason": "Error during verdict synthesis",
            "sources": [],
            "claim": claim,
            "calibration_note": f"Exception raised: {str(e)}"
        }
