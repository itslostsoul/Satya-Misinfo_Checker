"""
Image Forensics CLI Tool.

Command-line interface to analyze images for manipulation, AI-generation,
deepfakes, and doctored screenshot chyrons.

Usage:
  python cli.py <image_path_or_url> [options]

Examples:
  python cli.py sample.jpg
  python cli.py https://example.com/breaking_news.png --screenshot
  python cli.py spliced.png --save-ela ela_heatmap.jpg
  python cli.py test.jpg --json
"""

import argparse
import io
import json
import os
import sys
from typing import Optional
import urllib.request
from PIL import Image

from forensics.fusion import analyze_image_forensics
from forensics.ela import compute_ela

# Safe stdout UTF-8 reconfiguration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GRAY = "\033[90m"
WHITE = "\033[97m"


def format_bar(score: float, width: int = 20) -> str:
    """Generates an ASCII/Unicode progress bar for confidence visualization."""
    filled = int(round(score * width))
    bar = "=" * filled + "." * (width - filled)
    return f"[{bar}] {score:.0%}"


def load_image(source: str) -> Image.Image:
    """Loads image from local filesystem path or remote HTTP(S) URL."""
    if source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(
            source,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageForensics/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            return Image.open(io.BytesIO(data))
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        return Image.open(source)


def print_report(source: str, result: dict, verbose: bool = False) -> None:
    """Prints a styled terminal forensics report."""
    verdict = result["verdict"]
    conf = result["confidence"]
    reason = result["reason"]
    signals = result["signals"]

    # Pick verdict styling
    if verdict == "manipulated":
        verdict_badge = f"{BOLD}{RED}[!] MANIPULATED / SYNTHETIC{RESET}"
        score_color = RED
    elif verdict == "authentic":
        verdict_badge = f"{BOLD}{GREEN}[OK] AUTHENTIC / NATURAL{RESET}"
        score_color = GREEN
    else:
        verdict_badge = f"{BOLD}{YELLOW}[?] UNCERTAIN / INCONCLUSIVE{RESET}"
        score_color = YELLOW

    print("\n" + "=" * 65)
    print(f"{BOLD}{CYAN}IMAGE FORENSICS REPORT{RESET}")
    print("=" * 65)
    print(f"Target Source  : {source}")
    print(f"Verdict        : {verdict_badge}")
    print(f"Confidence     : {score_color}{format_bar(conf)}{RESET}")
    print(f"Explanation    : {WHITE}{reason}{RESET}")
    print("-" * 65)

    print(f"{BOLD}Detector Breakdown:{RESET}")

    # 1. ELA & Splicing
    ela = signals.get("ela", {})
    spatial = ela.get("spatial_anomaly_score", 0.0)
    anom_blocks = ela.get("anomalous_blocks", 0)
    ela_status = f"{RED}Anomalous ({anom_blocks} blocks){RESET}" if spatial > 0.65 else f"{GREEN}Uniform{RESET}"
    print(f"  * {BOLD}Error Level Analysis (ELA):{RESET} {format_bar(spatial, 14)} [{ela_status}]")

    # 2. AI Generation
    ai_sig = signals.get("ai_generator", {})
    ai_conf = ai_sig.get("ai_confidence", 0.0)
    ai_model = ai_sig.get("model_used", "fallback")
    ai_status = f"{RED}Synthetic{RESET}" if ai_sig.get("is_ai_generated") else f"{GREEN}Natural{RESET}"
    print(f"  * {BOLD}AI Generator Classifier  :{RESET} {format_bar(ai_conf, 14)} [{ai_status}] {GRAY}({ai_model}){RESET}")

    # 3. Metadata
    meta = signals.get("metadata", {})
    meta_tools = meta.get("editing_tools", []) + meta.get("genai_tools", [])
    if meta_tools:
        meta_str = f"{RED}Signatures Detected ({', '.join(meta_tools)}){RESET}"
    elif meta.get("is_stripped"):
        meta_str = f"{GRAY}EXIF Stripped{RESET}"
    else:
        meta_str = f"{GREEN}Camera EXIF Clean{RESET}"
    print(f"  * {BOLD}Metadata Provenance      :{RESET} {meta_str}")

    # 4. Deepfake
    df = signals.get("deepfake", {})
    if df.get("face_detected"):
        df_score = df.get("deepfake_score", 0.0)
        df_status = f"{RED}Manipulated{RESET}" if df.get("is_deepfake") else f"{GREEN}Natural Face{RESET}"
        print(f"  * {BOLD}Face Deepfake Detector   :{RESET} {format_bar(df_score, 14)} [{df_status}] ({df.get('face_count', 1)} face(s))")
    else:
        print(f"  * {BOLD}Face Deepfake Detector   :{RESET} {GRAY}Skipped (No faces detected){RESET}")

    # 5. Chyron & Screenshot
    chyron = signals.get("chyron_tampering", {})
    if chyron.get("is_screenshot"):
        ch_score = chyron.get("tamper_score", 0.0)
        ch_status = f"{RED}Doctored Banner{RESET}" if ch_score > 0.4 else f"{GREEN}Consistent{RESET}"
        print(f"  * {BOLD}Screenshot / Chyron      :{RESET} {format_bar(ch_score, 14)} [{ch_status}] (Aspect: {chyron.get('aspect_ratio')})")

    if verbose:
        print("\n" + "-" * 65)
        print(f"{BOLD}Detailed Signals (Debug JSON):{RESET}")
        print(json.dumps(signals, indent=2))

    print("=" * 65 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Image Forensics Engineer CLI: Detect AI generation, ELA splicing, deepfakes, and chyron doctoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Path to local image file or HTTP/HTTPS image URL")
    parser.add_argument("--screenshot", action="store_true", help="Flag image explicitly as a screenshot")
    parser.add_argument("--claimed-source", help="Claimed publisher URL for headline cross-referencing")
    parser.add_argument("--save-ela", metavar="OUT_PATH", help="Save the amplified ELA difference heatmap to disk")
    parser.add_argument("--json", action="store_true", help="Output raw JSON to stdout")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose signal debugging output")

    args = parser.parse_args()

    try:
        image = load_image(args.image)
    except Exception as e:
        print(f"{RED}Error loading image:{RESET} {e}", file=sys.stderr)
        return 1

    # Optional: Save ELA heatmap
    if args.save_ela:
        try:
            ela_img, _ = compute_ela(image)
            ela_img.save(args.save_ela)
            if not args.json:
                print(f"{GREEN}Saved ELA heatmap to:{RESET} {args.save_ela}")
        except Exception as e:
            print(f"{RED}Failed to save ELA heatmap:{RESET} {e}", file=sys.stderr)

    # Run complete analysis
    result = analyze_image_forensics(
        image=image,
        claimed_source_url=args.claimed_source,
        force_screenshot=args.screenshot
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(args.image, result, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
