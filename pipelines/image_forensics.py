import time
from PIL import Image
from forensics.fusion import analyze_image_forensics
from schema import PipelineResult, Verdict

async def check_image_manipulation(image_path: str) -> PipelineResult:
    """
    Runs the advanced image forensics pipeline on the specified image.
    Fuses ELA, AI detector, deepfake flags, and metadata signatures.
    """
    start_time = time.monotonic()
    try:
        img = Image.open(image_path)
        img.load()
        res = analyze_image_forensics(img)
        
        # Map forensics verdict to schema.Verdict
        verdict_map = {
            "manipulated": Verdict.LIKELY_FALSE,
            "authentic": Verdict.LIKELY_TRUE,
            "uncertain": Verdict.UNVERIFIABLE
        }
        verdict = verdict_map.get(res.get("verdict"), Verdict.UNVERIFIABLE)
        
        evidence = [res.get("reason", "No details available")]
        
        latency = int((time.monotonic() - start_time) * 1000)
        return PipelineResult(
            pipeline_name="image_forensics",
            verdict=verdict,
            confidence=res.get("confidence", 0.0),
            evidence=evidence,
            sources=[],
            latency_ms=latency
        )
    except Exception as e:
        latency = int((time.monotonic() - start_time) * 1000)
        return PipelineResult(
            pipeline_name="image_forensics",
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.0,
            evidence=[],
            latency_ms=latency,
            error=str(e)
        )
