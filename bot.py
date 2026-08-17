"""
OWNER: Person 1 - Bot Integration & Orchestrator

Entry point. Wires Telegram -> pipelines -> card -> reply, and enforces
the 60-second budget end-to-end.

Setup:
  1. Message @BotFather on Telegram, /newbot, grab the token.
  2. Put it in a .env file: TELEGRAM_BOT_TOKEN=xxxx
  3. pip install -r requirements.txt
  4. python bot.py
"""

import asyncio
import logging
import os
import tempfile
import time

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from card import format_telegram_message, merge_results
from pipelines.image_forensics import check_image_manipulation
from pipelines.reverse_context import check_reverse_context
from pipelines.text_claim import check_text_claim
from schema import PipelineResult, Verdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIME_BUDGET_SECONDS = 60


async def run_with_timeout(coro, name: str, timeout: float) -> PipelineResult:
    """Wrap a pipeline call so one slow/broken pipeline can't blow the 60s budget."""
    start = time.monotonic()
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - deliberately broad: any pipeline failure degrades gracefully
        logger.exception("pipeline %s failed", name)
        return PipelineResult(
            pipeline_name=name,
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.0,
            evidence=[],
            latency_ms=int((time.monotonic() - start) * 1000),
            error=str(e),
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    start = time.monotonic()
    await update.message.chat.send_action(ChatAction.TYPING)

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        image_path = tmp.name

    caption = update.message.caption or ""

    remaining = TIME_BUDGET_SECONDS - (time.monotonic() - start)
    results = await asyncio.gather(
        run_with_timeout(check_image_manipulation(image_path), "image_forensics", remaining),
        run_with_timeout(check_reverse_context(image_path, caption), "reverse_context", remaining),
    )

    os.unlink(image_path)

    card = merge_results(list(results))
    await update.message.reply_text(format_telegram_message(card), parse_mode="Markdown")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    start = time.monotonic()
    await update.message.chat.send_action(ChatAction.TYPING)

    remaining = TIME_BUDGET_SECONDS - (time.monotonic() - start)
    result = await run_with_timeout(check_text_claim(update.message.text), "text_claim", remaining)

    card = merge_results([result])
    await update.message.reply_text(format_telegram_message(card), parse_mode="Markdown")


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set — copy .env.example to .env and fill it in")
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    # TODO (stretch): add a voice handler once pipelines/voice.py is implemented

    logger.info("Forward-checker bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
