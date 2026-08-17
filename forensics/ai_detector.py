"""
AI Generation Detection Module.

Integrates Hugging Face open checkpoints (e.g. `umm-maybe/AI-image-detector`,
`Organika/sdxl-detector`, or `Falconsai/ai_detector`) with:
1. Lazy loading & in-memory caching.
2. Device auto-detection (CUDA / Apple MPS / CPU).
3. 2D FFT Spectral & Texture Artifact Heuristics (complementary & fallback signal).
4. Graceful fallback when offline or in minimal environments.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default open model from Hugging Face
DEFAULT_MODEL_NAME = "umm-maybe/AI-image-detector"


def analyze_spectral_artifacts(image: Image.Image) -> Dict[str, Any]:
    """
    Analyzes 2D Fourier Power Spectrum for high-frequency periodic grid artifacts,
    a characteristic fingerprint of diffusion and GAN upsampling layers.

    Args:
        image: PIL Image input.

    Returns:
        Dictionary with spectral anomaly score and frequency domain metrics.
    """
    gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)

    # 2D Fast Fourier Transform and shift zero-frequency to center
    f_transform = np.fft.fft2(arr)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-6)

    # Calculate radial energy distribution
    cy, cx = 128, 128
    y, x = np.ogrid[:256, :256]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    high_freq_mask = (r > 64) & (r < 120)
    high_freq_energy = float(np.mean(magnitude_spectrum[high_freq_mask]))
    center_energy = float(np.mean(magnitude_spectrum[r <= 32])) + 1e-6

    high_to_low_ratio = high_freq_energy / center_energy

    # Check for periodic grid spikes in high frequencies
    high_freq_vals = magnitude_spectrum[high_freq_mask]
    hf_std = float(np.std(high_freq_vals))
    hf_peak_ratio = (float(np.max(high_freq_vals)) - high_freq_energy) / (hf_std + 1e-6)

    # Spectral score: higher means more anomalous high-frequency grid repetition
    spectral_score = min(1.0, max(0.0, (high_to_low_ratio - 0.5) * 1.5 + (hf_peak_ratio - 2.5) * 0.15))

    return {
        "spectral_artifact_score": round(float(spectral_score), 4),
        "high_to_low_ratio": round(high_to_low_ratio, 4),
        "high_freq_peak_ratio": round(hf_peak_ratio, 4),
    }


class AIGenerationDetector:
    """
    Wraps Hugging Face image classification model with lazy loading and fallback.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._pipeline = None
        self._load_failed = False
        self._failure_reason: Optional[str] = None
        self._device = None

    def _determine_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load_model(self) -> bool:
        """Lazy load model pipeline if not already loaded."""
        if self._pipeline is not None:
            return True
        if self._load_failed:
            return False

        try:
            from transformers import pipeline
            self._device = self._determine_device()
            logger.info("Loading AI image detector model '%s' on %s...", self.model_name, self._device)
            device_id = 0 if self._device == "cuda" else -1
            self._pipeline = pipeline("image-classification", model=self.model_name, device=device_id)
            logger.info("AI image detector loaded successfully.")
            return True
        except Exception as e:
            self._load_failed = True
            self._failure_reason = str(e)
            logger.warning("Could not load Hugging Face model '%s' (%s). Using fallback heuristics.", self.model_name, e)
            return False

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs AI-generation detection on the input image.

        Args:
            image: PIL Image.

        Returns:
            Dictionary with ai_confidence, is_ai_generated, model_used, and signals.
        """
        spectral_data = analyze_spectral_artifacts(image)
        spectral_score = spectral_data["spectral_artifact_score"]
        evidence: List[str] = []

        is_loaded = self._load_model()

        if is_loaded and self._pipeline is not None:
            try:
                # Run Hugging Face model
                predictions = self._pipeline(image)
                # Parse predictions (typical labels: 'artificial', 'fake', 'ai-generated', 'real', 'human')
                ai_score = 0.0
                top_label = "unknown"
                for pred in predictions:
                    label = str(pred.get("label", "")).lower()
                    score = float(pred.get("score", 0.0))
                    if any(w in label for w in ["artificial", "fake", "ai", "synthetic", "generated"]):
                        ai_score = max(ai_score, score)
                    elif any(w in label for w in ["real", "human", "natural", "authentic"]):
                        ai_score = max(ai_score, 1.0 - score)

                # Fuse with spectral heuristics
                fused_ai_score = (ai_score * 0.85) + (spectral_score * 0.15)
                is_ai = fused_ai_score > 0.60

                if is_ai:
                    evidence.append(f"AI generation detector model ({self.model_name}) confidence: {fused_ai_score:.0%}")
                else:
                    evidence.append(f"AI generation detector model ({self.model_name}) classifies as natural ({(1.0 - fused_ai_score):.0%})")

                return {
                    "is_ai_generated": is_ai,
                    "ai_confidence": round(fused_ai_score, 4),
                    "raw_model_score": round(ai_score, 4),
                    "model_used": self.model_name,
                    "device": self._device,
                    "status": "model_inferred",
                    "spectral_signals": spectral_data,
                    "evidence": evidence,
                }
            except Exception as e:
                logger.warning("Inference with model %s failed: %s. Falling back.", self.model_name, e)

        # Fallback mode: Spectral artifact + high-frequency distribution heuristics
        fallback_score = spectral_score
        is_ai_fallback = fallback_score > 0.65

        if is_ai_fallback:
            evidence.append(f"Spectral analysis detected periodic high-frequency generator artifacts ({fallback_score:.0%})")
        else:
            evidence.append("Frequency spectrum consistent with natural optical capture")

        return {
            "is_ai_generated": is_ai_fallback,
            "ai_confidence": round(fallback_score, 4),
            "raw_model_score": None,
            "model_used": "ela_fallback",
            "fallback_reason": self._failure_reason or "transformers_not_loaded",
            "status": "fallback",
            "spectral_signals": spectral_data,
            "evidence": evidence,
        }


# Global singleton instance for easy import
ai_detector = AIGenerationDetector()


def detect_ai_generation(image: Image.Image) -> Dict[str, Any]:
    """Convenience function for AI-generation detection."""
    return ai_detector.detect(image)
