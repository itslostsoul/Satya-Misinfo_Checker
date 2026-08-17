import time
from schema import PipelineResult, Verdict

async def check_voice(audio_path: str) -> PipelineResult:
    """
    Analyzes voice records for AI voice clones (stub).
    """
    start_time = time.monotonic()
    latency = int((time.monotonic() - start_time) * 1000)
    return PipelineResult(
        pipeline_name="voice",
        verdict=Verdict.UNVERIFIABLE,
        confidence=0.0,
        evidence=["Voice authentication is not yet active on the server."],
        sources=[],
        latency_ms=latency
    )
