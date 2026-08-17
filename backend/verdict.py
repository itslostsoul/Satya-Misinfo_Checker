def merge_and_calibrate(pipeline_results: dict) -> dict:
    img = pipeline_results.get("image_result")
    txt = pipeline_results.get("text_result")

    signals = []

    if img:
        if img.get("is_ai_generated") and img.get("ai_confidence", 0) > 0.75:
            signals.append({
                "verdict": "false",
                "confidence": img["ai_confidence"] * 100,
                "weight": 0.6
            })
        elif img.get("date_mismatch") and img.get("delta_days", 0) > 180:
            signals.append({
                "verdict": "false",
                "confidence": 70,
                "weight": 0.5
            })

    if txt and txt.get("verdict") != "unverifiable" and txt.get("confidence", 0) > 0:
        signals.append({
            "verdict": txt["verdict"],
            "confidence": txt["confidence"],
            "weight": 0.7
        })

    if not signals:
        return {
            "verdict": "unverifiable",
            "confidence": None,
            "image_signal": img,
            "text_signal": txt
        }

    total_weight = sum(s["weight"] for s in signals)
    weighted_conf = sum(s["confidence"] * s["weight"] for s in signals) / total_weight

    false_weight = sum(s["weight"] for s in signals if s["verdict"] == "false")
    true_weight = sum(s["weight"] for s in signals if s["verdict"] == "true")
    dominant = "false" if false_weight >= true_weight else "true"

    has_direct_match = txt and txt.get("matched_article") is not None
    if dominant == "false" and weighted_conf > 85 and not has_direct_match:
        weighted_conf = 84

    return {
        "verdict": dominant,
        "confidence": round(weighted_conf),
        "image_signal": img,
        "text_signal": txt
    }
