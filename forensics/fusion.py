"""
Score Fusion and Decision Engine Module.

Combines forensic sub-signals into a calibrated, explainable verdict:
- ELA anomaly score & spatial clustering
- EXIF / IPTC / XMP metadata provenance & editing signatures
- AI-generation detector confidence & frequency domain artifacts
- Deepfake / facial boundary analysis (when faces are present)
- Doctored-screenshot & chyron tampering indicators

Outputs standardized schema:
{
  "verdict": "manipulated" | "authentic" | "uncertain",
  "confidence": 0.0-1.0,
  "reason": "short explanation, 1-2 sentences",
  "signals": { ... raw sub-scores from each detector, for debugging }
}
"""

from typing import Any, Dict, List, Optional
from PIL import Image

from forensics.ela import run_ela_pipeline
from forensics.ai_detector import detect_ai_generation
from forensics.deepfake import detect_deepfake
from forensics.chyron import detect_chyron_tampering


# ============================================================================
# CONFIGURABLE WEIGHT CONSTANTS (Easily tunable)
# ============================================================================
BASE_WEIGHT_AI_GENERATOR = 0.40   # AI generator model / spectral artifacts
BASE_WEIGHT_ELA_ANOMALY  = 0.25   # Spatial compression variance / splicing
BASE_WEIGHT_METADATA     = 0.15   # Editing software & Gen-AI provenance tags
BASE_WEIGHT_DEEPFAKE     = 0.20   # Face deepfake & blending boundary analysis
EXTRA_WEIGHT_CHYRON      = 0.20   # Doctored screenshot / chyron tampering

# Thresholds for verdict mapping
THRESHOLD_MANIPULATED = 0.75
THRESHOLD_UNCERTAIN_LOWER = 0.40


def fuse_scores(
    ela_res: Dict[str, Any],
    ai_res: Dict[str, Any],
    df_res: Dict[str, Any],
    chyron_res: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Dynamically normalizes weights based on detector availability and face presence,
    then combines sub-scores into a final calibrated manipulation score.

    Returns:
        Dictionary containing fused score, active weights, driving signals, and reasons.
    """
    signals: List[Dict[str, Any]] = []

    # 1. AI Generation signal
    ai_score = float(ai_res.get("ai_confidence", 0.0))
    signals.append({
        "name": "ai_generation",
        "score": ai_score,
        "base_weight": BASE_WEIGHT_AI_GENERATOR,
        "evidence": ai_res.get("evidence", [])
    })

    # 2. ELA Spatial Anomaly signal
    ela_score = float(ela_res.get("spatial_score", 0.0))
    signals.append({
        "name": "ela_anomaly",
        "score": ela_score,
        "base_weight": BASE_WEIGHT_ELA_ANOMALY,
        "evidence": [f"ELA spatial compression anomaly score: {ela_score:.2f}"]
    })

    # 3. Metadata Provenance signal
    meta_score = float(ela_res.get("metadata_score", 0.0))
    signals.append({
        "name": "metadata_provenance",
        "score": meta_score,
        "base_weight": BASE_WEIGHT_METADATA,
        "evidence": ela_res.get("metadata_details", {}).get("evidence", [])
    })

    # 4. Deepfake signal (only if face detected)
    if df_res.get("face_detected", False) and not df_res.get("skipped", True):
        df_score = float(df_res.get("deepfake_score", 0.0))
        signals.append({
            "name": "deepfake_face",
            "score": df_score,
            "base_weight": BASE_WEIGHT_DEEPFAKE,
            "evidence": df_res.get("evidence", [])
        })

    # 5. Chyron & Screenshot Tampering signal (if active)
    if chyron_res and chyron_res.get("is_screenshot", False):
        chyron_score = float(chyron_res.get("chyron_tamper_score", 0.0))
        signals.append({
            "name": "chyron_tampering",
            "score": chyron_score,
            "base_weight": EXTRA_WEIGHT_CHYRON,
            "evidence": chyron_res.get("evidence", [])
        })

    # Calculate normalized weighted score
    total_weight = sum(s["base_weight"] for s in signals)
    fused_score = sum(s["score"] * s["base_weight"] for s in signals) / max(0.01, total_weight)

    # If decisive ground-truth metadata tags exist (e.g. Photoshop or Midjourney explicit tag)
    meta_details = ela_res.get("metadata_details", {})
    if meta_details.get("genai_tools_detected"):
        fused_score = max(fused_score, 0.95)
    elif meta_details.get("editing_tools_detected"):
        fused_score = max(fused_score, 0.82)

    fused_score = min(1.0, max(0.0, fused_score))

    return {
        "fused_score": round(fused_score, 4),
        "active_signals": signals,
        "total_weight": round(total_weight, 3),
    }


def generate_reason(
    verdict: str,
    fused_score: float,
    ela_res: Dict[str, Any],
    ai_res: Dict[str, Any],
    df_res: Dict[str, Any],
    chyron_res: Optional[Dict[str, Any]]
) -> str:
    """
    Synthesizes a concise 1-2 sentence human-readable explanation highlighting
    the top driving forensic signals.
    """
    drivers: List[str] = []

    meta_details = ela_res.get("metadata_details", {})
    gen_tools = meta_details.get("genai_tools_detected", [])
    edit_tools = meta_details.get("editing_tools_detected", [])

    if gen_tools:
        drivers.append(f"embedded Generative AI signatures ({', '.join(gen_tools)})")
    elif edit_tools:
        drivers.append(f"editing software tags ({', '.join(edit_tools)})")

    # High AI model confidence
    ai_conf = ai_res.get("ai_confidence", 0.0)
    if ai_res.get("is_ai_generated") and ai_conf > 0.65:
        model_name = ai_res.get("model_used", "AI detector")
        drivers.append(f"high synthetic generation probability ({ai_conf:.0%}) from {model_name}")

    # ELA spatial anomaly
    spatial_score = ela_res.get("spatial_score", 0.0)
    if spatial_score > 0.65:
        anom_blocks = ela_res.get("spatial_details", {}).get("anomalous_blocks_count", 0)
        drivers.append(f"localized ELA compression anomalies across {anom_blocks} spliced block regions")

    # Deepfake facial artifacts
    if df_res.get("face_detected") and df_res.get("is_deepfake"):
        drivers.append("facial boundary blending seams and texture smoothing disparities")

    # Chyron tampering
    if chyron_res and chyron_res.get("is_screenshot") and chyron_res.get("chyron_tamper_score", 0) > 0.4:
        drivers.append("mismatched text rendering sharpness in lower-third chyron banner")

    # Construct final explanation based on verdict
    if verdict == "manipulated":
        if drivers:
            reasons_joined = " and ".join(drivers[:2])
            return f"Image appears manipulated or synthetic based on {reasons_joined}."
        return f"Image exhibits multiple compression and generative manipulation anomalies (score: {fused_score:.0%})."

    elif verdict == "authentic":
        clean_indicators = []
        if spatial_score < 0.35:
            clean_indicators.append("uniform compression history across grid blocks")
        if not df_res.get("face_detected"):
            clean_indicators.append("natural optical sensor characteristics")
        elif not df_res.get("is_deepfake"):
            clean_indicators.append("consistent facial feature lighting and textures")

        indicators_str = " and ".join(clean_indicators) if clean_indicators else "natural camera provenance"
        return f"Image appears authentic with {indicators_str} and no detected manipulation seams."

    else:  # uncertain
        if drivers:
            return f"Verification is inconclusive: detected weak signals of {drivers[0]}, but insufficient evidence to confirm full manipulation."
        return "Image shows mixed forensic indicators; compression artifacts cannot be definitively distinguished from harmless re-saving."


def analyze_image_forensics(
    image: Image.Image,
    claimed_source_url: Optional[str] = None,
    force_screenshot: bool = False,
    ela_quality: int = 90
) -> Dict[str, Any]:
    """
    End-to-end Image Forensics Pipeline.
    Executes ELA, AI detection, Deepfake detection, Chyron tampering, and Score Fusion.

    Returns:
        Standardized dictionary:
        {
            "verdict": "manipulated" | "authentic" | "uncertain",
            "confidence": 0.0-1.0,
            "reason": "1-2 sentence human explanation",
            "signals": { ... raw sub-scores from each detector }
        }
    """
    # 1. Run Baseline ELA + Metadata analysis
    ela_result = run_ela_pipeline(image, quality=ela_quality)

    # 2. Run AI Generation detection
    ai_result = detect_ai_generation(image)

    # 3. Run Face & Deepfake detection (gates deepfake analysis)
    deepfake_result = detect_deepfake(image)

    # 4. Run Doctored Screenshot / Chyron tampering analysis
    chyron_result = detect_chyron_tampering(
        image,
        claimed_source_url=claimed_source_url,
        force_screenshot=force_screenshot
    )

    # 5. Fuse scores
    fusion_output = fuse_scores(
        ela_res=ela_result,
        ai_res=ai_result,
        df_res=deepfake_result,
        chyron_res=chyron_result
    )
    fused_score = fusion_output["fused_score"]

    # 6. Map to verdict buckets
    if fused_score > THRESHOLD_MANIPULATED:
        verdict = "manipulated"
        confidence = fused_score
    elif fused_score < THRESHOLD_UNCERTAIN_LOWER:
        verdict = "authentic"
        confidence = 1.0 - fused_score
    else:
        verdict = "uncertain"
        # In uncertain range, confidence is distance from either decision boundary
        dist_to_center = abs(fused_score - 0.575)
        confidence = round(0.50 + (dist_to_center * 1.5), 3)

    confidence = round(min(0.99, max(0.40, confidence)), 3)

    # 7. Generate reason
    reason = generate_reason(
        verdict=verdict,
        fused_score=fused_score,
        ela_res=ela_result,
        ai_res=ai_result,
        df_res=deepfake_result,
        chyron_res=chyron_result
    )

    # Assemble comprehensive signals dictionary for debugging
    signals = {
        "fusion_score": fused_score,
        "thresholds": {
            "manipulated_gt": THRESHOLD_MANIPULATED,
            "uncertain_range": [THRESHOLD_UNCERTAIN_LOWER, THRESHOLD_MANIPULATED],
            "authentic_lt": THRESHOLD_UNCERTAIN_LOWER
        },
        "ela": {
            "spatial_anomaly_score": ela_result["spatial_score"],
            "anomalous_blocks": ela_result["spatial_details"]["anomalous_blocks_count"],
            "mean_difference": ela_result["ela_metrics"]["mean_difference"],
            "peak_variance_ratio": ela_result["spatial_details"]["peak_variance_ratio"]
        },
        "metadata": {
            "metadata_score": ela_result["metadata_score"],
            "is_stripped": ela_result["metadata_details"]["is_stripped"],
            "editing_tools": ela_result["metadata_details"]["editing_tools_detected"],
            "genai_tools": ela_result["metadata_details"]["genai_tools_detected"],
            "timestamp_inconsistent": ela_result["metadata_details"]["timestamp_inconsistent"]
        },
        "ai_generator": {
            "ai_confidence": ai_result["ai_confidence"],
            "is_ai_generated": ai_result["is_ai_generated"],
            "model_used": ai_result["model_used"],
            "status": ai_result["status"],
            "spectral_artifact_score": ai_result["spectral_signals"]["spectral_artifact_score"]
        },
        "deepfake": {
            "face_detected": deepfake_result["face_detected"],
            "face_count": deepfake_result["face_count"],
            "skipped": deepfake_result["skipped"],
            "deepfake_score": deepfake_result["deepfake_score"]
        },
        "chyron_tampering": {
            "is_screenshot": chyron_result["is_screenshot"],
            "aspect_ratio": chyron_result["aspect_ratio_match"],
            "tamper_score": chyron_result["chyron_tamper_score"],
            "ocr_text": chyron_result["ocr_details"]["text"][:100]
        }
    }

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
        "signals": signals
    }


class ScoreFusionEngine:
    """Wrapper class for score fusion pipeline."""

    def analyze(
        self,
        image: Image.Image,
        claimed_source_url: Optional[str] = None,
        force_screenshot: bool = False
    ) -> Dict[str, Any]:
        return analyze_image_forensics(
            image=image,
            claimed_source_url=claimed_source_url,
            force_screenshot=force_screenshot
        )
