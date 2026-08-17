import json
import logging
import os
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

# Load env variables (key: ANTHROPIC_API_KEY)
load_dotenv()

logger = logging.getLogger(__name__)

# Use the specific Claude model requested
MODEL_NAME = "claude-sonnet-4-6"

async def extract_claim(text: str) -> dict:
    """
    Extracts the core factual claim from a WhatsApp forward using Anthropic Claude.
    
    Args:
        text (str): The raw input text.
        
    Returns:
        dict: The structured JSON response containing:
            - claim (str or None): The clean, neutral, one-sentence claim.
            - original_language (str): "en", "hi", or "mixed".
            - claim_type (str): statistic, event, quote, policy, scientific, or other.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment")
        return {
            "claim": None,
            "original_language": "mixed",
            "claim_type": "other"
        }
        
    client = AsyncAnthropic(api_key=api_key)
    
    system_prompt = (
        "You are an expert system that extracts the core factual claim from rambling text messages or WhatsApp forwards.\n"
        "Your task is to identify the single most verifiable factual claim buried in the text.\n\n"
        "Rules:\n"
        "1. Strip all emotional language, political framing, religious framing, and forwarded headers (such as \"Forwarded as received\", \"Please share\", etc.).\n"
        "2. Return ONLY a clean, neutral, one-sentence factual claim in English.\n"
        "3. If no verifiable claim exists (e.g. pure opinion, personal prayer, jokes, wishes, greetings), set the \"claim\" field to null.\n"
        "4. Output must be a valid JSON object with the following fields:\n"
        "   - \"claim\": The extracted one-sentence claim, or null.\n"
        "   - \"original_language\": The language of the original text (\"en\", \"hi\", or \"mixed\").\n"
        "   - \"claim_type\": The type of claim (\"statistic\", \"event\", \"quote\", \"policy\", \"scientific\", or \"other\"). If claim is null, set this to \"other\".\n"
        "5. Do not output any markdown formatting, no code block backticks (like ```json), no preamble, and no explanation. Output ONLY the JSON string."
    )
    
    try:
        response = await client.messages.create(
            model=MODEL_NAME,
            max_tokens=400,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Text to analyze:\n\n{text}"
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
            
        parsed_result = json.loads(raw_output)
        return parsed_result
        
    except Exception as e:
        logger.exception("Error extracting claim via Anthropic client")
        return {
            "claim": None,
            "original_language": "mixed",
            "claim_type": "other"
        }
