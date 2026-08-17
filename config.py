"""
Central place to read API keys from the environment. Every pipeline
imports from here instead of calling os.getenv itself, so:
  - it's obvious at a glance which keys the project needs
  - a missing key degrades that one pipeline to "unverifiable"
    instead of crashing the whole bot
"""

import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# LLM (claim extraction, verdict summarization, translation)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

# Reverse image search
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Fact-check source lookup (Google Programmable Search Engine, scoped to
# PIB Fact Check / AltNews / BOOM). Optional — text_claim.py falls back
# to an empty source list without it.
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
