import time
from datetime import date, datetime
from backend.pipelines.image_pipeline import reverse_image_search
from schema import PipelineResult, Verdict, Source

async def check_reverse_context(image_path: str, caption: str) -> PipelineResult:
    """
    Performs a Google reverse image search using the image bytes.
    Detects when images are recycled out-of-context for misinformation.
    """
    start_time = time.monotonic()
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        serp_result = await reverse_image_search(image_bytes)
        
        earliest_date = serp_result.get("earliest_date")
        date_mismatch = False
        delta_days = 0
        evidence = []
        sources = []
        
        if earliest_date:
            try:
                parsed = datetime.strptime(earliest_date, "%Y-%m-%d").date()
                delta_days = (date.today() - parsed).days
                date_mismatch = delta_days > 30
                evidence.append(f"Image has been online since {earliest_date} ({delta_days} days ago).")
            except Exception:
                evidence.append(f"Image was previously seen online on {earliest_date}.")
        else:
            evidence.append("No historical records found for this image via reverse image search.")

        context = serp_result.get("context", "")
        if context:
            evidence.append(f"Found context matching original use: '{context}'")
            
        for url in serp_result.get("source_urls", []):
            sources.append(Source(title="Reverse Image Match", url=url, publisher="Google Reverse Image"))

        # If there is a date mismatch, it means it's likely false context (recycled media)
        if date_mismatch:
            verdict = Verdict.LIKELY_FALSE
            confidence = min(0.95, 0.5 + (delta_days / 365) * 0.1)
        else:
            verdict = Verdict.UNVERIFIABLE
            confidence = 0.0

        latency = int((time.monotonic() - start_time) * 1000)
        return PipelineResult(
            pipeline_name="reverse_context",
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            sources=sources,
            latency_ms=latency
        )
    except Exception as e:
        latency = int((time.monotonic() - start_time) * 1000)
        return PipelineResult(
            pipeline_name="reverse_context",
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.0,
            evidence=[],
            latency_ms=latency,
            error=str(e)
        )
