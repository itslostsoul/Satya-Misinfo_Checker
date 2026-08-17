"""
Shared verdict contract — every pipeline (image, text-claim, voice) must
return a dict matching PipelineResult, and the orchestrator merges them
into a single VerdictCard before rendering.

Agree on this shape as a team FIRST. Everyone builds against it in parallel.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    LIKELY_TRUE = "likely_true"
    LIKELY_FALSE = "likely_false"
    UNVERIFIABLE = "unverifiable"  # first-class output, not a fallback


@dataclass
class Source:
    title: str
    url: str
    publisher: str  # e.g. "AltNews", "BOOM", "PIB Fact Check"


@dataclass
class PipelineResult:
    """What each pipeline (image / text-claim / voice) returns."""
    pipeline_name: str          # "image_forensics" | "reverse_context" | "text_claim" | "voice"
    verdict: Verdict
    confidence: float           # 0.0-1.0, calibrated — don't just hardcode 0.9
    evidence: list[str]         # short bullet reasons, e.g. "image indexed since 2013-06-14"
    sources: list[Source] = field(default_factory=list)
    latency_ms: int = 0
    error: Optional[str] = None  # set if pipeline failed/timed out; orchestrator should degrade gracefully


@dataclass
class VerdictCard:
    """Final merged output rendered back into the chat."""
    verdict: Verdict
    confidence: float
    explanation_en: str          # <= 2 lines, plain language
    explanation_regional: str    # <= 2 lines, same content, regional language
    sources: list[Source]
    total_latency_ms: int
    blind_spots: Optional[str] = None  # e.g. "could not check audio" — surface honestly, don't hide gaps
