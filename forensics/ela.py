"""
Error Level Analysis (ELA) and Metadata Forensics Module.

Performs:
1. Re-compression difference analysis (ELA) at fixed JPEG quality.
2. Spatial grid variance analysis to detect localized spliced/inserted patches.
3. EXIF, IPTC, and XMP metadata inspection for photo editing (Photoshop, GIMP)
   and Generative AI signatures (Midjourney, Stable Diffusion, DALL-E, etc.).
"""

import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags


# Editing software signatures to flag
EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "canva", "paint.net", "pixlr",
    "photopea", "affinity", "snapseed", "coreldraw", "capture one",
    "vsco", "picsart", "afterlight", "fotor", "befunky"
]

# Generative AI signatures to flag
GENAI_SOFTWARE_KEYWORDS = [
    "midjourney", "stable diffusion", "stablediffusion", "dall-e", "dalle",
    "fooocus", "comfyui", "automatic1111", "novelai", "adobe firefly",
    "firefly", "bing image creator", "flux", "ideogram", "imagen",
    "dreamstudio", "leonardo.ai", "artbreeder", "craiyon"
]


def compute_ela(
    image: Image.Image,
    quality: int = 90,
    scale: int = 20
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Computes Error Level Analysis (ELA) by re-saving image as JPEG at target quality
    and calculating the amplified absolute pixel differences.

    Args:
        image: Source PIL Image.
        quality: JPEG compression quality (default: 90).
        scale: Amplification factor for visualizing differences (default: 20).

    Returns:
        Tuple of (amplified_ela_image, metrics_dict).
    """
    # Ensure RGB (handling transparency cleanly by compositing on white background)
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        orig = background
    else:
        orig = image.convert("RGB")

    # Re-save to memory at target quality
    buffer = io.BytesIO()
    orig.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    # Compute difference
    diff = ImageChops.difference(orig, resaved)

    # Convert difference to numpy array for numerical metrics
    diff_arr = np.asarray(diff, dtype=np.float32)
    mean_diff = float(np.mean(diff_arr)) if diff_arr.size > 0 else 0.0
    max_diff = float(np.max(diff_arr)) if diff_arr.size > 0 else 0.0
    std_diff = float(np.std(diff_arr)) if diff_arr.size > 0 else 0.0

    # Amplify difference for visual analysis
    extrema = diff.getextrema()
    max_val = max([ex[1] for ex in extrema]) if extrema else 1
    scale_factor = scale if max_val == 0 else min(scale, int(255.0 / max(max_val, 1)))
    scale_factor = max(1, scale_factor)

    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(scale_factor)

    metrics = {
        "mean_difference": round(mean_diff, 4),
        "max_difference": round(max_diff, 4),
        "std_difference": round(std_diff, 4),
        "scale_factor": scale_factor,
        "quality_tested": quality,
    }

    return ela_image, metrics


def analyze_ela_spatial(
    ela_image: Image.Image,
    block_size: int = 16
) -> Dict[str, Any]:
    """
    Divides ELA image into grid blocks and analyzes variance disparities.
    Spliced or inserted regions display significant localized error variance
    differing from the uniform background compression pattern.

    Args:
        ela_image: Amplified ELA PIL image.
        block_size: Size of square blocks in pixels (default: 16).

    Returns:
        Dictionary with spatial anomaly score, peak block coordinates, and stats.
    """
    gray_ela = ela_image.convert("L")
    arr = np.asarray(gray_ela, dtype=np.float32)
    height, width = arr.shape

    if height < block_size or width < block_size:
        return {
            "spatial_anomaly_score": 0.0,
            "anomalous_blocks_count": 0,
            "total_blocks": 0,
            "background_median_variance": 0.0,
            "peak_variance": 0.0,
            "peak_variance_ratio": 1.0,
            "anomalous_regions": [],
        }

    # Grid block statistics
    n_blocks_y = height // block_size
    n_blocks_x = width // block_size

    block_variances = []
    block_means = []
    block_coords = []

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y1 = by * block_size
            y2 = y1 + block_size
            x1 = bx * block_size
            x2 = x1 + block_size

            patch = arr[y1:y2, x1:x2]
            var = float(np.var(patch))
            mean_val = float(np.mean(patch))

            block_variances.append(var)
            block_means.append(mean_val)
            block_coords.append((x1, y1, block_size, block_size))

    if not block_variances:
        return {
            "spatial_anomaly_score": 0.0,
            "anomalous_blocks_count": 0,
            "total_blocks": 0,
            "background_median_variance": 0.0,
            "peak_variance": 0.0,
            "peak_variance_ratio": 1.0,
            "anomalous_regions": [],
        }

    variances_arr = np.array(block_variances, dtype=np.float32)
    means_arr = np.array(block_means, dtype=np.float32)

    bg_mean_var = float(np.median(variances_arr))
    bg_std_var = float(np.std(variances_arr)) + 1e-6

    # Threshold for anomaly: blocks exceeding median + 2.5 * std
    anomaly_threshold = bg_mean_var + (2.5 * bg_std_var)
    anomalous_mask = variances_arr > anomaly_threshold
    anomalous_count = int(np.sum(anomalous_mask))

    anomalous_regions = []
    for idx, is_anomaly in enumerate(anomalous_mask):
        if is_anomaly:
            x, y, w, h = block_coords[idx]
            anomalous_regions.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "variance": round(float(variances_arr[idx]), 2),
                "mean_brightness": round(float(means_arr[idx]), 2)
            })

    total_blocks = max(1, len(block_variances))
    anomaly_ratio = anomalous_count / total_blocks
    peak_variance = float(np.max(variances_arr)) if len(variances_arr) > 0 else 0.0
    peak_ratio = peak_variance / (bg_mean_var + 1.0)

    # Score calculation: weighted combo of peak ratio and localized cluster ratio
    # 0.0 = completely uniform compression, 1.0 = heavy localized splicing
    score = min(1.0, (anomaly_ratio * 3.5) + min(0.6, (peak_ratio - 1.0) / 10.0))
    score = max(0.0, score)

    return {
        "spatial_anomaly_score": round(score, 4),
        "anomalous_blocks_count": anomalous_count,
        "total_blocks": total_blocks,
        "background_median_variance": round(bg_mean_var, 2),
        "peak_variance": round(peak_variance, 2),
        "peak_variance_ratio": round(peak_ratio, 2),
        "anomalous_regions": anomalous_regions[:10],  # Top 10 for inspectability
    }


def extract_metadata_signals(image: Image.Image) -> Dict[str, Any]:
    """
    Extracts and scans EXIF/XMP metadata for signs of manipulation,
    photo-editing software tags, generative AI provenance, and timestamp discrepancies.

    Args:
        image: Source PIL image.

    Returns:
        Structured metadata signals dict with detected tools and anomaly score.
    """
    metadata: Dict[str, Any] = {}
    evidence: List[str] = []
    editing_tools_found: List[str] = []
    genai_tools_found: List[str] = []
    is_stripped = True
    timestamps: Dict[str, Optional[str]] = {
        "DateTime": None,
        "DateTimeOriginal": None,
        "DateTimeDigitized": None
    }

    try:
        exif_raw = image._getexif()  # type: ignore[attr-defined]
        if exif_raw:
            is_stripped = False
            for tag_id, val in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                metadata[tag_name] = str(val)

                if tag_name in timestamps:
                    timestamps[tag_name] = str(val)

                # Search software and description strings
                val_str = str(val).lower()
                for tool in EDITING_SOFTWARE_KEYWORDS:
                    if tool in val_str and tool not in editing_tools_found:
                        editing_tools_found.append(tool)
                for gen_tool in GENAI_SOFTWARE_KEYWORDS:
                    if gen_tool in val_str and gen_tool not in genai_tools_found:
                        genai_tools_found.append(gen_tool)

    except Exception:
        pass

    # Check PNG / WebP / JPEG info dictionary (e.g. text chunks where Stable Diffusion / DALL-E write prompts)
    for key, val in (image.info or {}).items():
        val_str = str(val).lower()
        key_str = str(key).lower()
        metadata[f"info_{key}"] = str(val)[:300]
        is_stripped = False

        for tool in EDITING_SOFTWARE_KEYWORDS:
            if tool in val_str and tool not in editing_tools_found:
                editing_tools_found.append(tool)
        for gen_tool in GENAI_SOFTWARE_KEYWORDS:
            if (gen_tool in val_str or gen_tool in key_str) and gen_tool not in genai_tools_found:
                genai_tools_found.append(gen_tool)

        # Check for prompt parameter blocks (common in ComfyUI / Automatic1111 / Midjourney)
        if any(marker in val_str for marker in ["parameters", "prompt:", "negative prompt:", "steps:", "sampler:"]):
            if "prompt_metadata" not in genai_tools_found:
                genai_tools_found.append("prompt_metadata")

    # Timestamp consistency check
    timestamp_inconsistent = False
    valid_ts = [t for t in timestamps.values() if t]
    if len(valid_ts) > 1 and len(set(valid_ts)) > 1:
        timestamp_inconsistent = True
        evidence.append(f"Inconsistent EXIF timestamps: {timestamps}")

    # Build evidence and calculate metadata score
    metadata_score = 0.0

    if genai_tools_found:
        metadata_score = 0.95
        evidence.append(f"Generative AI metadata signatures detected: {', '.join(genai_tools_found)}")
    elif editing_tools_found:
        metadata_score = 0.80
        evidence.append(f"Photo-editing software tags found: {', '.join(editing_tools_found)}")
    elif timestamp_inconsistent:
        metadata_score = 0.60
    elif is_stripped:
        metadata_score = 0.35
        evidence.append("EXIF metadata is stripped or absent (common in web/social forwards)")
    else:
        metadata_score = 0.10
        software = metadata.get("Software")
        make = metadata.get("Make", "")
        model = metadata.get("Model", "")
        if make or model:
            evidence.append(f"Camera hardware metadata present: {make} {model}".strip())
        elif software:
            evidence.append(f"Standard metadata present: {software}")

    return {
        "metadata_score": round(metadata_score, 4),
        "is_stripped": is_stripped,
        "editing_tools_detected": editing_tools_found,
        "genai_tools_detected": genai_tools_found,
        "timestamp_inconsistent": timestamp_inconsistent,
        "timestamps": timestamps,
        "evidence": evidence,
        "raw_tag_count": len(metadata),
        "sample_tags": {k: metadata[k] for k in list(metadata.keys())[:8]},
    }


def run_ela_pipeline(
    image: Image.Image,
    quality: int = 90
) -> Dict[str, Any]:
    """
    Executes full standalone ELA and metadata analysis pipeline.

    Args:
        image: PIL Image input.
        quality: JPEG compression quality for ELA (default: 90).

    Returns:
        Structured result dictionary with anomaly score, sub-scores, and reasoning.
    """
    ela_img, ela_metrics = compute_ela(image, quality=quality)
    spatial_results = analyze_ela_spatial(ela_img)
    meta_results = extract_metadata_signals(image)

    spatial_score = spatial_results["spatial_anomaly_score"]
    meta_score = meta_results["metadata_score"]

    # Fused ELA + Metadata score (pure baseline)
    if meta_results["genai_tools_detected"] or meta_results["editing_tools_detected"]:
        combined_score = max(meta_score, spatial_score)
    else:
        combined_score = (spatial_score * 0.65) + (meta_score * 0.35)

    reasons: List[str] = []
    if meta_results["evidence"]:
        reasons.extend(meta_results["evidence"])

    if spatial_score > 0.65:
        reasons.append(
            f"High localized ELA variance anomaly detected (score: {spatial_score:.2f}, "
            f"{spatial_results['anomalous_blocks_count']} anomalous blocks)"
        )
    elif spatial_score > 0.40:
        reasons.append(f"Moderate compression variance observed (score: {spatial_score:.2f})")
    else:
        reasons.append(f"Uniform compression profile across image grid (score: {spatial_score:.2f})")

    return {
        "ela_anomaly_score": round(combined_score, 4),
        "spatial_score": spatial_score,
        "metadata_score": meta_score,
        "ela_metrics": ela_metrics,
        "spatial_details": spatial_results,
        "metadata_details": meta_results,
        "reasons": reasons,
        "ela_image": ela_img,
    }


class ELAScanner:
    """Class wrapper for Error Level Analysis and Metadata scanning."""

    def __init__(self, quality: int = 90):
        self.quality = quality

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        return run_ela_pipeline(image, quality=self.quality)
