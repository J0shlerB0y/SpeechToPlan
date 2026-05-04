"""Обработка голосовых и аудио-сообщений."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from aiogram import F, Router
from aiogram.types import Message

from bot.config import BotConfig
from bot.services.pipeline import AssistantPipeline
from bot.utils.formatting import format_task

logger = logging.getLogger(__name__)
voice_router = Router(name="voice")


@voice_router.message(F.voice | F.audio)
async def handle_voice(
    message: Message,
    pipeline: AssistantPipeline,
    config: BotConfig,
) -> None:
    voice = message.voice or message.audio
    if voice.duration and voice.duration > config.max_audio_seconds:
        await message.answer(f"Аудио длиннее {config.max_audio_seconds} секунд — не обработаю.")
        return

    await message.chat.do("record_voice")
    file = await message.bot.get_file(voice.file_id)
    suffix = ".ogg" if message.voice else (Path(voice.file_name or "a.bin").suffix or ".bin")
    local_path = config.tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    try:
        await message.bot.download_file(file.file_path, destination=local_path)
        logger.info("Скачали аудио %s (%.1fс) -> %s", voice.file_id, voice.duration or 0, local_path)
        task = await pipeline.process_voice(local_path)
    except Exception:
        logger.exception("Voice pipeline failed")
        await message.answer("Не смог распознать аудио. Попробуй ещё раз или отправь текстом.")
        return
    finally:
        local_path.unlink(missing_ok=True)

    await message.answer(format_task(task), parse_mode="Markdown")
