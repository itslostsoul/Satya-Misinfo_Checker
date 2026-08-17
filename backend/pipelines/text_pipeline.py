import os
import httpx
from google import genai
from sentence_transformers import SentenceTransformer, util

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None
_embedder = None


def get_embedder():
    # Loaded lazily (not at import time) so the server can boot — and mock
    # mode can serve requests — without pulling ~470MB from HF Hub first.
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _embedder

async def run_text_pipeline(text: str) -> dict:
    claim = await extract_claim(text)
    articles = await search_fact_check_sources(claim)

    if not articles:
        return {
            "verdict": "unverifiable",
            "confidence": 0,
            "matched_article": None,
            "sources": []
        }

    embedder = get_embedder()
    claim_embedding = embedder.encode(claim, convert_to_tensor=True)
    best_match = None
    best_score = 0.0

    for article in articles:
        article_embedding = embedder.encode(
            article.get("text", article.get("title", "")),
            convert_to_tensor=True
        )
        score = float(util.cos_sim(claim_embedding, article_embedding))
        if score > best_score:
            best_score = score
            best_match = article

    if best_score < 0.5:
        return {"verdict": "unverifiable", "confidence": 0, "matched_article": None, "sources": []}

    return {
        "verdict": best_match.get("verdict", "unverifiable"),
        "confidence": int(best_score * 100),
        "matched_article": best_match,
        "sources": [{"title": best_match.get("title", "Source"), "url": best_match.get("url", "#")}]
    }

async def extract_claim(text: str) -> str:
    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=(
            "Extract the single core factual claim from this WhatsApp forward. "
            "Output only the claim in one English sentence. No explanation. No preamble.\n\n"
            f"Forward: {text}"
        )
    )
    return response.text.strip()

async def search_fact_check_sources(claim: str) -> list:
    GFC_KEY = os.getenv("GOOGLE_FACT_CHECK_KEY")
    if not GFC_KEY:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={"query": claim, "key": GFC_KEY, "languageCode": "en"}
            )
        return parse_gfc_response(resp.json())
    except Exception as e:
        print(f"[WARN] Fact check search failed: {e}")
        return []

def parse_gfc_response(data: dict) -> list:
    articles = []
    for item in data.get("claims", []):
        for review in item.get("claimReview", []):
            rating = review.get("textualRating", "").lower()
            verdict = "false"
            if any(w in rating for w in ["true", "correct", "accurate"]):
                verdict = "true"
            elif any(w in rating for w in ["false", "incorrect", "fake", "misleading"]):
                verdict = "false"
            else:
                verdict = "unverifiable"

            articles.append({
                "title": review.get("title", item.get("text", "Fact Check")),
                "url": review.get("url", "#"),
                "text": item.get("text", ""),
                "verdict": verdict
            })
    return articles
