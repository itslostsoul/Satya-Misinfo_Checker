"""
Standalone tester for reverse image search — deliberately does NOT import
pipelines/image_pipeline.py, because that file also loads the deepfake
detector (transformers/torch) at import time, which you don't need for
this and don't want to install just to test a SerpAPI call.

The two functions below are a direct copy of reverse_image_search() /
parse_serp_response() from image_pipeline.py. If those change over there,
update here too (or ask to factor them into a shared, transformers-free
module — cleaner long-term fix, not needed to unblock testing right now).

Usage (run from inside backend/, venv activated):
    python test_reverse_image_search.py path/to/image.jpg
"""

import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()  # picks up ../.env (SERPAPI_API_KEY)


async def reverse_image_search(image_bytes: bytes) -> dict:
    SERP_KEY = os.getenv("SERPAPI_API_KEY")
    if not SERP_KEY:
        print("SERPAPI_API_KEY not set in .env")
        return {}

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://serpapi.com/search",
            params={"engine": "google_reverse_image", "api_key": SERP_KEY},
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        )
    print(f"[raw] HTTP {resp.status_code}")
    print(resp.text[:1000])
    resp.raise_for_status()
    return parse_serp_response(resp.json())


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
        "source_urls": source_urls,
    }


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_reverse_image_search.py <image_path>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        image_bytes = f.read()

    try:
        result = await reverse_image_search(image_bytes)
    except httpx.HTTPStatusError as e:
        print(f"\nRequest failed: {e}")
        return

    print("\nresult:", result)


if __name__ == "__main__":
    asyncio.run(main())
