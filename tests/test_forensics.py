"""
Unit & Integration Test Suite for Image Forensics Pipeline.
"""

import io
import unittest
from PIL import Image, ImageDraw, PngImagePlugin

from forensics.ela import compute_ela, analyze_ela_spatial, extract_metadata_signals, run_ela_pipeline
from forensics.ai_detector import analyze_spectral_artifacts, detect_ai_generation
from forensics.deepfake import detect_faces, analyze_face_artifacts, detect_deepfake
from forensics.chyron import is_screenshot_geometry, detect_ui_chrome, analyze_chyron_tampering, cross_check_claimed_source, detect_chyron_tampering
from forensics.fusion import analyze_image_forensics, fuse_scores, generate_reason
from api import app
from starlette.testclient import TestClient


class TestELAForensics(unittest.TestCase):
    def setUp(self):
        # 1. Clean uniform image
        self.clean_img = Image.new("RGB", (256, 256), color=(200, 200, 200))

        # 2. Spliced image with an inserted high-contrast pattern
        self.spliced_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        draw = ImageDraw.Draw(self.spliced_img)
        draw.rectangle([64, 64, 192, 192], fill=(20, 40, 180))
        draw.line([64, 64, 192, 192], fill=(255, 255, 0), width=4)

    def test_compute_ela_metrics(self):
        ela_img, metrics = compute_ela(self.clean_img, quality=90)
        self.assertEqual(ela_img.size, (256, 256))
        self.assertIn("mean_difference", metrics)
        self.assertIn("max_difference", metrics)
        self.assertIn("scale_factor", metrics)
        self.assertGreaterEqual(metrics["scale_factor"], 1)

    def test_spatial_anomaly_clean_vs_spliced(self):
        clean_ela, _ = compute_ela(self.clean_img)
        clean_spatial = analyze_ela_spatial(clean_ela)

        spliced_ela, _ = compute_ela(self.spliced_img)
        spliced_spatial = analyze_ela_spatial(spliced_ela)

        self.assertGreater(
            spliced_spatial["spatial_anomaly_score"],
            clean_spatial["spatial_anomaly_score"],
            "Spliced image must have higher spatial anomaly score than clean image"
        )
        self.assertGreater(spliced_spatial["anomalous_blocks_count"], 0)

    def test_metadata_extraction_ai_and_editing_tags(self):
        img = Image.new("RGB", (100, 100), color=(120, 120, 120))
        # Add PNG info chunk with Midjourney / Photoshop marker
        img.info["parameters"] = "Prompt: A futuristic city in cyberpunk style. Steps: 30, Sampler: Euler a"
        img.info["Software"] = "Adobe Photoshop 2024"

        meta_res = extract_metadata_signals(img)
        self.assertIn("prompt_metadata", meta_res["genai_tools_detected"])
        self.assertIn("photoshop", meta_res["editing_tools_detected"])
        self.assertGreaterEqual(meta_res["metadata_score"], 0.80)


class TestAIGenerationDetector(unittest.TestCase):
    def test_spectral_artifacts(self):
        img = Image.new("RGB", (256, 256), color=(150, 150, 150))
        res = analyze_spectral_artifacts(img)
        self.assertIn("spectral_artifact_score", res)
        self.assertIn("high_to_low_ratio", res)
        self.assertGreaterEqual(res["spectral_artifact_score"], 0.0)
        self.assertLessEqual(res["spectral_artifact_score"], 1.0)

    def test_ai_detector_graceful_execution(self):
        img = Image.new("RGB", (128, 128), color=(80, 120, 160))
        res = detect_ai_generation(img)
        self.assertIn("ai_confidence", res)
        self.assertIn("is_ai_generated", res)
        self.assertIn("model_used", res)
        self.assertIn("status", res)
        self.assertIsInstance(res["evidence"], list)


class TestDeepfakeDetector(unittest.TestCase):
    def test_skip_when_no_face_present(self):
        img_no_face = Image.new("RGB", (100, 100), color=(10, 50, 90))
        res = detect_deepfake(img_no_face)
        self.assertFalse(res["face_detected"])
        self.assertTrue(res["skipped"])
        self.assertEqual(res["face_count"], 0)
        self.assertIn("skipped", res["reason"].lower())

    def test_face_detection_and_artifact_analysis(self):
        # Synthetic face oval
        img_face = Image.new("RGB", (300, 300), color=(40, 40, 40))
        draw = ImageDraw.Draw(img_face)
        draw.ellipse([80, 50, 220, 240], fill=(220, 180, 140))  # Skin tone oval
        res = detect_deepfake(img_face)
        self.assertTrue(res["face_detected"])
        self.assertFalse(res["skipped"])
        self.assertGreaterEqual(res["face_count"], 1)
        self.assertIn("deepfake_score", res)


class TestChyronTampering(unittest.TestCase):
    def test_screenshot_geometry(self):
        is_scr, conf, tag = is_screenshot_geometry(1080, 1920)
        self.assertTrue(is_scr)
        self.assertEqual(tag, "9:16")
        self.assertGreater(conf, 0.8)

        is_scr_desk, _, tag_desk = is_screenshot_geometry(1920, 1080)
        self.assertTrue(is_scr_desk)
        self.assertEqual(tag_desk, "16:9")

    def test_claimed_source_matching(self):
        matched = cross_check_claimed_source("BREAKING: BBC News reports election results", "https://www.bbc.com/news")
        self.assertTrue(matched["domain_found_in_banner"])
        self.assertEqual(matched["match_status"], "verified_domain_match")

        mismatched = cross_check_claimed_source("BREAKING: Unrelated story from local blog", "https://reuters.com")
        self.assertFalse(mismatched["domain_found_in_banner"])


class TestScoreFusion(unittest.TestCase):
    def test_end_to_end_forensics_pipeline(self):
        img = Image.new("RGB", (200, 200), color=(180, 180, 180))
        result = analyze_image_forensics(img)

        self.assertIn("verdict", result)
        self.assertIn(result["verdict"], ["manipulated", "authentic", "uncertain"])
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIn("reason", result)
        self.assertIsInstance(result["reason"], str)
        self.assertGreater(len(result["reason"]), 10)
        self.assertIn("signals", result)

    def test_manipulated_verdict_when_spliced_with_editing_tags(self):
        # Create heavily manipulated image with editing tag
        img = Image.new("RGB", (300, 300), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 250, 250], fill=(10, 20, 190))
        img.info["Software"] = "Adobe Photoshop 2024"

        result = analyze_image_forensics(img)
        self.assertEqual(result["verdict"], "manipulated")
        self.assertGreaterEqual(result["confidence"], 0.75)
        self.assertIn("photoshop", result["reason"].lower())


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")

    def test_info_endpoint(self):
        resp = self.client.get("/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("detectors", data)
        self.assertGreaterEqual(len(data["detectors"]), 4)

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("docs_url", data)

    def test_analyze_file_upload(self):
        img = Image.new("RGB", (150, 150), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = self.client.post(
            "/analyze",
            files={"image_file": ("test.jpg", buf, "image/jpeg")},
            data={"is_screenshot": "false"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("verdict", data)
        self.assertIn("confidence", data)
        self.assertIn("reason", data)
        self.assertIn("signals", data)

    def test_verify_forward_endpoint_with_image(self):
        img = Image.new("RGB", (100, 100), color=(150, 150, 150))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = self.client.post(
            "/api/verify",
            files={"image": ("test.jpg", buf, "image/jpeg")},
            data={"language": "tamil"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("verdict", data)
        self.assertIn("explanation_en", data)
        self.assertIn("sources", data)

    def test_image_modes_and_edge_dimensions(self):
        # Test RGBA, LA, P, Grayscale, CMYK
        for mode in ["RGBA", "LA", "P", "L", "CMYK"]:
            img = Image.new(mode, (64, 64))
            res = analyze_image_forensics(img)
            self.assertIn(res["verdict"], ["manipulated", "authentic", "uncertain"])

        # Test tiny edge-case dimensions
        for size in [(1, 1), (2, 2), (5, 5), (15, 15)]:
            img = Image.new("RGB", size, (100, 100, 100))
            res = analyze_image_forensics(img)
            self.assertIn(res["verdict"], ["manipulated", "authentic", "uncertain"])


if __name__ == "__main__":
    unittest.main()
