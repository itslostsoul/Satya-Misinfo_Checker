"""
Image Forensics Package.

Provides multi-layer image authenticity and manipulation detection:
- ELA (Error Level Analysis) & EXIF/Metadata Forensics
- AI-Generation & Synthetic Image Detection
- Face & Deepfake Forensics
- Doctored Screenshot & Chyron Tampering Forensics
- Calibrated Multi-Signal Score Fusion Engine
"""

from forensics.ela import (
    compute_ela,
    analyze_ela_spatial,
    extract_metadata_signals,
    run_ela_pipeline,
    ELAScanner,
)
from forensics.ai_detector import (
    detect_ai_generation,
    analyze_spectral_artifacts,
    AIGenerationDetector,
)
from forensics.deepfake import (
    detect_faces,
    analyze_face_artifacts,
    detect_deepfake,
    DeepfakeDetector,
)
from forensics.chyron import (
    is_screenshot_geometry,
    detect_ui_chrome,
    extract_ocr_text,
    analyze_chyron_tampering,
    cross_check_claimed_source,
    detect_chyron_tampering,
    ChyronDetector,
)
from forensics.fusion import (
    analyze_image_forensics,
    fuse_scores,
    generate_reason,
    ScoreFusionEngine,
)

__all__ = [
    "analyze_image_forensics",
    "compute_ela",
    "analyze_ela_spatial",
    "extract_metadata_signals",
    "run_ela_pipeline",
    "ELAScanner",
    "detect_ai_generation",
    "analyze_spectral_artifacts",
    "AIGenerationDetector",
    "detect_faces",
    "analyze_face_artifacts",
    "detect_deepfake",
    "DeepfakeDetector",
    "is_screenshot_geometry",
    "detect_ui_chrome",
    "extract_ocr_text",
    "analyze_chyron_tampering",
    "cross_check_claimed_source",
    "detect_chyron_tampering",
    "ChyronDetector",
    "fuse_scores",
    "generate_reason",
    "ScoreFusionEngine",
]
