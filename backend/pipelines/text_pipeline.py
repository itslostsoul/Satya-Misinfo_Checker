import os
import httpx
from anthropic import AsyncAnthropic
from sentence_transformers import SentenceTransformer, util

anthropic_client = None

def get_anthropic_client():
    global anthropic_client
    if anthropic_client is None:
        anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return anthropic_client

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

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
    client = get_anthropic_client()
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                "Extract the single core factual claim from this WhatsApp forward. "
                "Output only the claim in one English sentence. No explanation. No preamble.\n\n"
                f"Forward: {text}"
            )
        }]
    )
    return response.content[0].text.strip()

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
