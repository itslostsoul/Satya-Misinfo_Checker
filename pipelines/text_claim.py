import time
from backend.pipelines.text_pipeline import run_text_pipeline
from schema import PipelineResult, Verdict, Source

async def check_text_claim(text: str) -> PipelineResult:
    """
    Analyzes text claims, extracts core assertions via Claude, and cross-checks
    them against multilingual fact-checking databases.
    """
    start_time = time.monotonic()
    try:
        res = await run_text_pipeline(text)
        
        # Map verdict
        v_map = {
            "true": Verdict.LIKELY_TRUE,
            "false": Verdict.LIKELY_FALSE,
            "unverifiable": Verdict.UNVERIFIABLE
        }
        verdict = v_map.get(res.get("verdict"), Verdict.UNVERIFIABLE)
        
        evidence = []
        matched = res.get("matched_article")
        if matched:
            evidence.append(f"Matched fact check: '{matched.get('title')}' rated as '{matched.get('verdict')}'")
        else:
            evidence.append("No matching fact checks found in GFC databases.")

        sources = []
        for s in res.get("sources", []):
            sources.append(Source(title=s.get("title", "Source"), url=s.get("url", "#"), publisher="Google Fact Check"))

        latency = int((time.monotonic() - start_time) * 1000)
        return PipelineResult(
            pipeline_name="text_claim",
            verdict=verdict,
            confidence=res.get("confidence", 0) / 100.0,
            evidence=evidence,
            sources=sources,
            latency_ms=latency
        )
    except Exception as e:
        latency = int((time.monotonic() - start_time) * 1000)
        return PipelineResult(
            pipeline_name="text_claim",
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.0,
            evidence=[],
            latency_ms=latency,
            error=str(e)
        )
