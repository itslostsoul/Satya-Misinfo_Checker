import logging
import urllib.parse
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# The Press Information Bureau (PIB) Fact Check page (https://pib.gov.in/factcheck.aspx)
# does not expose a standard keyword-based GET search endpoint.
# Scraping this page is a best-effort attempt that fetches the main landing page and
# searches for links containing keywords from the claim.
async def scrape_pib(claim: str, client: httpx.AsyncClient) -> list[dict]:
    """Scrapes PIB Fact Check landing page for matching keyword links."""
    results = []
    try:
        url = "https://pib.gov.in/factcheck.aspx"
        resp = await client.get(url, headers=HEADERS, timeout=10.0)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        keywords = [w.lower() for w in claim.split() if len(w) > 3]
        if not keywords:
            keywords = [claim.lower()]
            
        for a in soup.find_all("a"):
            text = a.get_text(strip=True)
            href = a.get("href")
            if not href or not text:
                continue
                
            if any(k in text.lower() for k in keywords):
                full_url = urllib.parse.urljoin(url, href)
                results.append({
                    "title": text,
                    "url": full_url,
                    "snippet": "Fact check matching claim keywords found on PIB Fact Check page.",
                    "source": "PIB",
                    "published_date": None
                })
                if len(results) >= 3:
                    break
    except Exception as e:
        logger.exception("Error scraping PIB Fact Check page")
        raise e
    return results

# AltNews search endpoint (https://www.altnews.in/?s={query}) is a confirmed queryable
# WordPress search endpoint. We perform a search query and parse the resulting article cards.
async def scrape_altnews(claim: str, client: httpx.AsyncClient) -> list[dict]:
    """Scrapes AltNews for matching fact check articles."""
    results = []
    try:
        query_encoded = urllib.parse.quote(claim)
        url = f"https://www.altnews.in/?s={query_encoded}"
        resp = await client.get(url, headers=HEADERS, timeout=10.0)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.find_all("article", class_="card")
        
        for art in articles:
            title_el = art.select_one("h3.card__title a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            article_url = title_el.get("href")
            
            content_type_el = art.select_one(".card__content-type")
            content_type = content_type_el.get_text(strip=True) if content_type_el else "Fact Check"
            verdict_el = art.select_one(".verdict-badge")
            verdict = verdict_el.get_text(strip=True) if verdict_el else ""
            
            snippet = f"[{content_type}] Verdict: {verdict}" if verdict else content_type
            
            date_el = art.select_one("time.card__date")
            published_date = date_el.get("datetime") or date_el.get_text(strip=True) if date_el else None
            
            results.append({
                "title": title,
                "url": article_url,
                "snippet": snippet,
                "source": "AltNews",
                "published_date": published_date
            })
            if len(results) >= 3:
                break
    except Exception as e:
        logger.exception("Error scraping AltNews search results")
        raise e
    return results

# BOOM search endpoint (https://www.boomlive.in/search?q={query}) is a confirmed queryable
# custom search endpoint. We perform a search query and parse the resulting story items.
async def scrape_boom(claim: str, client: httpx.AsyncClient) -> list[dict]:
    """Scrapes BOOM Live for matching fact check stories."""
    results = []
    try:
        query_encoded = urllib.parse.quote(claim)
        url = f"https://www.boomlive.in/search?q={query_encoded}"
        resp = await client.get(url, headers=HEADERS, timeout=10.0)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.find_all(class_="boom-item3")
        
        for item in items:
            title_el = item.select_one("h4.font-alt a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            rel_url = title_el.get("href")
            article_url = urllib.parse.urljoin("https://www.boomlive.in", rel_url) if rel_url else ""
            
            cat_el = item.select_one(".category_name_link")
            category = cat_el.get_text(strip=True) if cat_el else "Fact Check"
            
            snippet = f"[{category}]"
            
            date_el = item.select_one(".date")
            published_date = date_el.get_text(strip=True) if date_el else None
            
            results.append({
                "title": title,
                "url": article_url,
                "snippet": snippet,
                "source": "BOOM",
                "published_date": published_date
            })
            if len(results) >= 3:
                break
    except Exception as e:
        logger.exception("Error scraping BOOM search results")
        raise e
    return results

async def search_fact_checks(claim: str) -> dict:
    """
    Search PIB, AltNews, and BOOM for fact-checks matching the claim.
    
    Args:
        claim (str): The factual claim to search.
        
    Returns:
        dict: The final response containing results, sources checked, and sources failed.
    """
    sources_checked = ["PIB", "AltNews", "BOOM"]
    sources_failed = []
    all_results = []
    seen_urls = set()
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        # 1. PIB
        try:
            pib_results = await scrape_pib(claim, client)
            for r in pib_results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        except Exception:
            sources_failed.append("PIB")
            
        # 2. AltNews
        try:
            altnews_results = await scrape_altnews(claim, client)
            for r in altnews_results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        except Exception:
            sources_failed.append("AltNews")
            
        # 3. BOOM
        try:
            boom_results = await scrape_boom(claim, client)
            for r in boom_results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        except Exception:
            sources_failed.append("BOOM")
            
    return {
        "results": all_results,
        "sources_checked": sources_checked,
        "sources_failed": sources_failed
    }
