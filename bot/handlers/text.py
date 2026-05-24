from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.types import Message

from bot.services.pipeline import AssistantPipeline
from bot.utils.formatting import format_task

logger = logging.getLogger(__name__)
text_router = Router(name="text")


@text_router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message, pipeline: AssistantPipeline) -> None:
    await message.chat.do("typing")
    try:
        task = await pipeline.process_text(message.text or "")
    except Exception:
        logger.exception("LLM пайплайн упал на тексте")
        await message.answer("Не получилось разобрать сообщение, попробуй переформулировать.")
        return
    await message.answer(format_task(task), parse_mode="Markdown")
