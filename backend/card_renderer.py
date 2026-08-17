import os
from google import genai
from anthropic import Anthropic

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None

async def render_card(verdict: dict, language: str) -> dict:
    v = verdict["verdict"]
    conf = verdict["confidence"]
    evidence_summary = build_evidence_summary(verdict)

    prompt = (
        f"You are Satya, a fact-checker for Indian WhatsApp users.\n\n"
        f"Verdict: {v.upper()}\n"
        f"Confidence: {conf if conf else 'N/A'}%\n"
        f"Evidence: {evidence_summary}\n\n"
        f"Write exactly two lines:\n"
        f"Line 1 (EN): Plain English, grandparent level, no jargon.\n"
        f"Line 2 ({language.upper()}): Same meaning in {language}, natural and colloquial.\n\n"
        f"Format strictly as:\n"
        f"EN: <english explanation>\n"
        f"{language.upper()}: <regional explanation>"
    )

    if anthropic_client:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text_response = response.content[0].text
    elif gemini_client:
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        text_response = response.text
    else:
        # Static fallback if no API keys are present
        text_response = f"EN: Claim is {v}.\n{language.upper()}: Claim is {v} in {language}."

    lines = text_response.strip().split('\n')
    en_line = lines[0].replace("EN: ", "").strip() if lines else ""
    regional_line = lines[1].replace(f"{language.upper()}: ", "").strip() if len(lines) > 1 else ""

    sources = []
    txt = verdict.get("text_signal")
    img = verdict.get("image_signal")

    if txt and txt.get("sources"):
        sources.extend(txt["sources"])
    if img and img.get("source_urls"):
        sources += [{"title": u, "url": u} for u in img["source_urls"][:2]]

    return {
        "verdict": v,
        "confidence": conf,
        "explanation_en": en_line,
        "explanation_regional": regional_line,
        "sources": sources[:3]
    }

def build_evidence_summary(verdict: dict) -> str:
    parts = []
    img = verdict.get("image_signal")
    txt = verdict.get("text_signal")

    if img:
        if img.get("is_ai_generated"):
            parts.append(f"Image detected as AI-generated ({int(img.get('ai_confidence', 0) * 100)}% confidence)")
        if img.get("date_mismatch"):
            parts.append(f"Image found online {img.get('delta_days', 0)} days ago — predates claimed date")
    if txt:
        if txt.get("matched_article"):
            parts.append(f"Fact-check match: {txt['matched_article'].get('title', 'Unknown source')}")
        elif txt.get("sources"):
            parts.append(f"Fact-check matches: {', '.join(s.get('title', 'Source') for s in txt['sources'][:2])}")

    return '; '.join(parts) if parts else "No direct evidence found"
