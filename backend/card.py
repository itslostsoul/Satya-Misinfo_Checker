"""
OWNER: Person 5 - Card / UX + Regional Language

Job:
1. Merge PipelineResults into one VerdictCard (simple rule-based merge below —
   feel free to make this smarter, but keep it explainable for the judges).
2. Render the card as plain-language text, English + regional language,
   <= 2 lines each, readable by a non-technical relative.
3. Own the "documented blind spots" deliverable — be honest about what
   the system can't catch.
"""

from anthropic import Anthropic

import config
from schema import PipelineResult, Source, Verdict, VerdictCard

REGIONAL_LANGUAGE = "Hindi"  # TODO: confirm as a team; Hindi is the safe default

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else None


def merge_results(results: list[PipelineResult]) -> VerdictCard:
    """Very simple merge: most confident non-unverifiable result wins;
    if all are unverifiable (or empty), the card is honestly unverifiable."""
    usable = [r for r in results if r.error is None]
    decisive = [r for r in usable if r.verdict != Verdict.UNVERIFIABLE]

    total_latency = sum(r.latency_ms for r in results)
    all_sources: list[Source] = [s for r in usable for s in r.sources]

    if decisive:
        best = max(decisive, key=lambda r: r.confidence)
        verdict = best.verdict
        confidence = best.confidence
    else:
        best = None
        verdict = Verdict.UNVERIFIABLE
        confidence = 0.0

    explanation_en = render_explanation_en(verdict, best, usable)
    explanation_regional = translate(explanation_en)  # TODO: real translation

    blind_spots = None
    failed = [r.pipeline_name for r in results if r.error is not None]
    if failed:
        blind_spots = f"Could not check: {', '.join(failed)}"

    return VerdictCard(
        verdict=verdict,
        confidence=confidence,
        explanation_en=explanation_en,
        explanation_regional=explanation_regional,
        sources=all_sources,
        total_latency_ms=total_latency,
        blind_spots=blind_spots,
    )


def render_explanation_en(verdict: Verdict, best: PipelineResult | None, all_results: list[PipelineResult]) -> str:
    if verdict == Verdict.UNVERIFIABLE:
        return "We could not confirm or deny this. Please don't forward it until it's verified."
    reason = best.evidence[0] if best and best.evidence else "no details available"
    label = "This looks TRUE." if verdict == Verdict.LIKELY_TRUE else "This looks FALSE."
    return f"{label} {reason}"


def translate(text_en: str) -> str:
    if _client is None:
        return text_en  # ANTHROPIC_API_KEY not set — passthrough placeholder

    response = _client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Translate this into simple, natural {REGIONAL_LANGUAGE} that a "
                "non-technical grandparent would understand. Reply with only "
                f"the translation.\n\n{text_en}"
            ),
        }],
    )
    return response.content[0].text.strip()


def format_telegram_message(card: VerdictCard) -> str:
    verdict_labels = {
        Verdict.LIKELY_TRUE: "✅ Likely TRUE",
        Verdict.LIKELY_FALSE: "❌ Likely FALSE",
        Verdict.UNVERIFIABLE: "❓ Unverifiable",
    }
    lines = [
        f"*{verdict_labels[card.verdict]}* (confidence: {card.confidence:.0%})",
        "",
        card.explanation_en,
        card.explanation_regional,
    ]
    if card.sources:
        lines.append("")
        lines.append("Sources:")
        lines += [f"- {s.publisher}: {s.url}" for s in card.sources]
    if card.blind_spots:
        lines.append("")
        lines.append(f"_Note: {card.blind_spots}_")
    lines.append("")
    lines.append(f"_Checked in {card.total_latency_ms}ms_")
    return "\n".join(lines)
