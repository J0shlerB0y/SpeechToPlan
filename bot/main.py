"""Точка входа Telegram-бота. Запуск: `python -m bot.main`."""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import load_config
from bot.handlers import common_router, text_router, voice_router
from bot.services.pipeline import AssistantPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("bot")


async def main() -> None:
    cfg = load_config()
    logger.info("Загружаем модели… (whisper=%s, llm=%s, quant=%s)",
                cfg.asr_model_size, cfg.llm_model_path, cfg.llm_quant)

    pipeline = await AssistantPipeline.build(cfg)
    logger.info("Модели в VRAM. Стартуем polling.")

    bot = Bot(cfg.token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp["pipeline"] = pipeline
    dp["config"] = cfg

    dp.include_router(common_router)
    dp.include_router(voice_router)
    dp.include_router(text_router)

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
