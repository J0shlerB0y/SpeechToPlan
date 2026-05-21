"""Тонкий фасад над HTTP-клиентом ML-сервиса.

Хендлеры бота продолжают работать с `pipeline.process_text()` / `process_voice()`,
не зная, что под капотом HTTP, а не локальные модели.
"""
from __future__ import annotations

import logging
from pathlib import Path

from bot.services.ml_client import MLClient
from shared.schemas import PlannerTask

logger = logging.getLogger(__name__)


class AssistantPipeline:
    def __init__(self, ml: MLClient) -> None:
        self.ml = ml

    @classmethod
    async def build(cls, cfg) -> "AssistantPipeline":
        ml = MLClient(base_url=cfg.ml_service_url, timeout=cfg.ml_request_timeout)
        # «прогревочный» вызов: убеждаемся, что ML-сервис вообще доступен
        try:
            status = await ml.health()
            logger.info("ML-сервис отвечает: %s", status)
        except Exception:
            logger.exception("ML-сервис недоступен по %s", cfg.ml_service_url)
            raise
        return cls(ml=ml)

    async def process_text(self, text: str) -> PlannerTask:
        return await self.ml.process_text(text)

    async def process_voice(self, audio_path: Path) -> PlannerTask:
        return await self.ml.process_voice(audio_path)

    async def close(self) -> None:
        await self.ml.close()
