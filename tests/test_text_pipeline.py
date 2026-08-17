import os
import sys
import pytest
from unittest.mock import AsyncMock, patch

# Mock ANTHROPIC_API_KEY for tests so the environment checks pass
os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-testing"

# Ensure the backend directory is in the path for importing pipelines
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipelines.claim_extractor import extract_claim
from pipelines.fact_checker import search_fact_checks
from pipelines.verdict_engine import synthesize_verdict
from orchestrator import analyze_text

# Define mock classes for Anthropic response
class MockContentBlock:
    def __init__(self, text):
        self.text = text

class MockMessage:
    def __init__(self, text):
        self.content = [MockContentBlock(text)]

class MockResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP Error")

# --- Unit Tests ---

@pytest.mark.asyncio
@patch("pipelines.claim_extractor.AsyncAnthropic")
async def test_claim_extractor_hindi_english(mock_anthropic):
    """
    Test claim extractor with sample mixed Hindi-English text.
    """
    mock_client = AsyncMock()
    mock_anthropic.return_value = mock_client
    
    mock_json_response = (
        '{"claim": "Government announces free ration for all till 2030", '
        '"original_language": "mixed", '
        '"claim_type": "policy"}'
    )
    mock_client.messages.create.return_value = MockMessage(mock_json_response)
    
    sample_text = "Forwarded as received: Govt announces free ration for all till 2030, PM Modi confirmed in press conference yesterday"
    result = await extract_claim(sample_text)
    
    assert result["claim"] == "Government announces free ration for all till 2030"
    assert result["original_language"] == "mixed"
    assert result["claim_type"] == "policy"
    mock_client.messages.create.assert_called_once()

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_fact_checker_mocks(mock_get):
    """
    Test fact checker with mocked HTTP responses for PIB, AltNews, and BOOM.
    """
    # Define mock response HTML content for each source
    pib_html = '<html><body><a href="https://pib.gov.in/factcheck-free-ration">Government announces free ration scheme details</a></body></html>'
    altnews_html = """
    <html><body>
        <article class="card">
            <h3 class="card__title">
                <a href="https://www.altnews.in/is-govt-announcing-free-ration-till-2030-fact-check/">Is government giving free ration till 2030? Fake claim viral</a>
            </h3>
            <span class="card__content-type">Fact Check</span>
            <span class="verdict-badge">False</span>
            <time class="card__date" datetime="2026-08-10T12:00:00+05:30">10th August 2026</time>
        </article>
    </body></html>
    """
    boom_html = """
    <html><body>
        <div class="boom-item3">
            <h4 class="font-alt"><a href="/fact-check/is-pm-modi-offering-free-ration-22900">PM Modi Free Ration Scheme Viral Claim Is Fake</a></h4>
            <a href="/fact-check" class="category_name_link">Fact Check</a>
            <span class="date">15 Aug 2026</span>
        </div>
    </body></html>
    """
    
    def side_effect(url, **kwargs):
        if "pib.gov.in" in url:
            return MockResponse(pib_html)
        elif "altnews.in" in url:
            return MockResponse(altnews_html)
        elif "boomlive.in" in url:
            return MockResponse(boom_html)
        return MockResponse("<html></html>")
        
    mock_get.side_effect = side_effect
    
    claim = "free ration for all till 2030"
    result = await search_fact_checks(claim)
    
    assert "results" in result
    results = result["results"]
    assert len(results) > 0
    
    # Assert that results contain entries from the sources
    sources = [r["source"] for r in results]
    assert "PIB" in sources or "AltNews" in sources or "BOOM" in sources

@pytest.mark.asyncio
@patch("pipelines.verdict_engine.AsyncAnthropic")
async def test_verdict_engine_low_confidence(mock_anthropic):
    """
    Test verdict engine calibration - low confidence should override verdict to unverifiable.
    """
    mock_client = AsyncMock()
    mock_anthropic.return_value = mock_client
    
    # Mocking a response with confidence < 0.6
    mock_json_response = (
        '{"verdict": "false", '
        '"confidence": 0.45, '
        '"reason": "The search results are somewhat relevant but not conclusive.", '
        '"sources": [{"title": "Fake Claim", "url": "https://altnews.in/fake", "source": "AltNews"}], '
        '"calibration_note": "Relevance is too low to say false definitively."}'
    )
    mock_client.messages.create.return_value = MockMessage(mock_json_response)
    
    claim = "free ration for all till 2030"
    fact_checks = [{"title": "Fake Claim", "url": "https://altnews.in/fake", "source": "AltNews", "snippet": "..."}]
    
    result = await synthesize_verdict("Sample context", claim, fact_checks)
    
    # Assert calibration override worked
    assert result["verdict"] == "unverifiable"
    assert "overridden" in result["calibration_note"].lower()
    assert result["confidence"] == 0.45

# --- Integration Tests ---

# Skip by default unless RUN_INTEGRATION env var is set to "true"
RUN_INTEGRATION = os.getenv("RUN_INTEGRATION", "false").lower() == "true"
integration_only = pytest.mark.skipif(not RUN_INTEGRATION, reason="Integration test skipped by default")

@pytest.mark.integration
@integration_only
@pytest.mark.asyncio
async def test_full_pipeline_integration():
    """
    Integration test that runs the full pipeline with live APIs.
    Requires ANTHROPIC_API_KEY to be set in environment/.env.
    """
    test_text = "Forwarded as received: Govt announces free ration for all till 2030, PM Modi confirmed in press conference yesterday"
    result = await analyze_text(test_text)
    
    assert "verdict" in result
    assert "confidence" in result
    assert "reason" in result
    assert "sources" in result
    assert result["claim"] is not None
