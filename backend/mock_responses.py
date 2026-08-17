import asyncio

MOCK_RESPONSES = [
    {
        "verdict": "false",
        "confidence": 78,
        "explanation_en": "This image was found online from 2018, not from yesterday as claimed.",
        "explanation_regional": "இந்த படம் 2018 ஆம் ஆண்டிலிருந்து எடுக்கப்பட்டது, நேற்று அல்ல.",
        "sources": [{"title": "AltNews (2018)", "url": "https://altnews.in"}]
    },
    {
        "verdict": "unverifiable",
        "confidence": None,
        "explanation_en": "No matching fact-check found. This claim cannot be verified with available evidence.",
        "explanation_regional": "இந்த தகவலை சரிபார்க்க போதுமான ஆதாரம் இல்லை.",
        "sources": []
    }
]

async def mock_verify(text, image) -> dict:
    await asyncio.sleep(1.5)
    # Return false mock if image uploaded, unverifiable for text-only
    return MOCK_RESPONSES[0] if image else MOCK_RESPONSES[1]
