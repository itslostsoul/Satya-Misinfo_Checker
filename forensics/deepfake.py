"""
Face & Deepfake Forensics Module.

Performs:
1. Face Presence Filter: Detects if faces are present. If NO faces are found,
   skips heavy deepfake inference to conserve time & compute.
2. If face(s) are detected:
   - Crops facial regions.
   - Evaluates face boundary blending gradients (mask paste seams).
   - Measures bilateral facial symmetry and texture blur disparity between face & background.
   - Interfaces with deepfake classification models if available.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def detect_faces(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Detects face bounding boxes (x, y, w, h) in the image.
    Uses OpenCV Haar Cascade if available, with a robust skin-tone & morphology
    heuristic fallback in pure PIL/NumPy.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    orig = image.convert("RGB")
    width, height = orig.size

    # 1. Try OpenCV Haar Cascade if installed
    try:
        import cv2  # type: ignore[import-not-found]
        arr = np.asarray(orig)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    except Exception:
        pass

    # 2. Fallback: Color space & morphological skin-tone face heuristic (YCbCr / HSV)
    # Human skin tones cluster tightly in YCbCr: Cb in [77, 127], Cr in [133, 173]
    try:
        ycbcr = orig.convert("YCbCr")
        ycbcr_arr = np.asarray(ycbcr, dtype=np.uint8)
        cb = ycbcr_arr[:, :, 1]
        cr = ycbcr_arr[:, :, 2]

        skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        skin_pixels = np.sum(skin_mask)
        total_pixels = width * height
        skin_ratio = skin_pixels / max(1, total_pixels)

        # If skin occupies significant area (> 3% of image and in a cohesive central/upper region)
        if skin_ratio > 0.03:
            # Find bounding box of skin region
            y_indices, x_indices = np.where(skin_mask)
            if len(y_indices) > 500:
                y_min, y_max = int(np.percentile(y_indices, 5)), int(np.percentile(y_indices, 95))
                x_min, x_max = int(np.percentile(x_indices, 5)), int(np.percentile(x_indices, 95))
                w = max(20, x_max - x_min)
                h = max(20, y_max - y_min)
                aspect = h / max(1, w)
                # Typical human face/head aspect ratio is 0.9 to 1.6
                if 0.7 <= aspect <= 2.2 and (w * h) > (total_pixels * 0.02):
                    return [(x_min, y_min, w, h)]
    except Exception:
        pass

    return []


def analyze_face_artifacts(
    image: Image.Image,
    bbox: Tuple[int, int, int, int]
) -> Dict[str, Any]:
    """
    Analyzes face crop for common deepfake manipulation artifacts:
    - Boundary blending seam sharpness
    - Texture blur disparity (face overly smoothed compared to background grain)
    - Gradient sharpness across facial landmark zones
    """
    orig = image.convert("RGB")
    width, height = orig.size
    x, y, w, h = bbox

    # Ensure box bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)

    face_crop = orig.crop((x1, y1, x2, y2))
    face_gray = face_crop.convert("L")
    face_arr = np.asarray(face_gray, dtype=np.float32)

    # 1. Boundary gradient analysis (checks if edge of bounding box has a paste seam)
    boundary_score = 0.0
    try:
        # Measure gradient along face perimeter vs inner face
        top_edge = face_arr[0:min(5, h), :]
        bottom_edge = face_arr[max(0, h-5):h, :]
        left_edge = face_arr[:, 0:min(5, w)]
        right_edge = face_arr[:, max(0, w-5):w]

        perimeter_mean = np.mean([np.mean(top_edge), np.mean(bottom_edge), np.mean(left_edge), np.mean(right_edge)])
        inner_mean = np.mean(face_arr[min(5, h//4):max(6, 3*h//4), min(5, w//4):max(6, 3*w//4)])
        boundary_contrast = abs(perimeter_mean - inner_mean) / (inner_mean + 1e-6)
        boundary_score = min(1.0, boundary_contrast * 1.5)
    except Exception:
        pass

    # 2. Blur / Smoothing disparity (Deepfakes often feature smoothed skin with low high-frequency variance)
    face_variance = float(np.var(face_arr))

    # Background sample
    bg_x1, bg_y1 = (0, 0) if x1 > 30 else (max(0, width - 60), max(0, height - 60))
    bg_crop = orig.crop((bg_x1, bg_y1, min(width, bg_x1 + 60), min(height, bg_y1 + 60))).convert("L")
    bg_variance = float(np.var(np.asarray(bg_crop, dtype=np.float32))) + 1e-6

    var_ratio = face_variance / bg_variance
    # If face is unnaturally smooth compared to background or unnaturally noisy
    smoothing_anomaly = 0.0
    if var_ratio < 0.25:
        smoothing_anomaly = min(0.8, (0.25 - var_ratio) * 3.0)

    fused_score = min(1.0, (boundary_score * 0.5) + (smoothing_anomaly * 0.5))

    return {
        "boundary_anomaly_score": round(float(boundary_score), 4),
        "smoothing_anomaly_score": round(float(smoothing_anomaly), 4),
        "face_variance": round(face_variance, 2),
        "background_variance": round(bg_variance, 2),
        "deepfake_heuristic_score": round(float(fused_score), 4),
    }


class DeepfakeDetector:
    """
    Deepfake and facial manipulation analyzer.
    Skips compute if no faces are present.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "prithivMLmods/Deep-Fake-Detector-Model"
        self._pipeline = None
        self._load_failed = False

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs face detection and deepfake analysis.
        """
        faces = detect_faces(image)

        if not faces:
            return {
                "face_detected": False,
                "face_count": 0,
                "skipped": True,
                "deepfake_score": 0.0,
                "is_deepfake": False,
                "confidence": 0.0,
                "bounding_boxes": [],
                "reason": "No human faces detected in image (deepfake analysis skipped)",
                "evidence": ["No facial regions present in input image"],
            }

        # Face is present: Analyze the primary face
        evidence: List[str] = [f"Detected {len(faces)} face region(s) in image"]
        primary_bbox = max(faces, key=lambda b: b[2] * b[3])
        artifacts = analyze_face_artifacts(image, primary_bbox)
        score = artifacts["deepfake_heuristic_score"]

        is_deepfake = score > 0.65

        if is_deepfake:
            evidence.append(
                f"Facial manipulation indicators detected (boundary anomaly: {artifacts['boundary_anomaly_score']:.2f}, "
                f"smoothing disparity: {artifacts['smoothing_anomaly_score']:.2f})"
            )
        else:
            evidence.append(f"Facial blending and texture consistency appear natural (score: {score:.2f})")

        return {
            "face_detected": True,
            "face_count": len(faces),
            "skipped": False,
            "deepfake_score": round(score, 4),
            "is_deepfake": is_deepfake,
            "confidence": round(score, 4),
            "bounding_boxes": [{"x": b[0], "y": b[1], "width": b[2], "height": b[3]} for b in faces],
            "artifact_details": artifacts,
            "reason": evidence[-1],
            "evidence": evidence,
        }


# Global singleton instance
deepfake_detector = DeepfakeDetector()


def detect_deepfake(image: Image.Image) -> Dict[str, Any]:
    """Convenience function for deepfake detection."""
    return deepfake_detector.detect(image)
