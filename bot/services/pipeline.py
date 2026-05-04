"""Тонкий слой-оркестратор: связывает asr_emotion + llm.

Бот не должен знать деталей моделей; он работает только с этим pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from asr_emotion.inference import AsrEmotionPipeline
from llm.inference import GemmaPlannerLLM
from shared.schemas import EnrichedUtterance, PlannerTask

logger = logging.getLogger(__name__)


class AssistantPipeline:
    """Один экземпляр — одна загрузка моделей в VRAM."""

    def __init__(self, asr: AsrEmotionPipeline, llm: GemmaPlannerLLM) -> None:
        self.asr = asr
        self.llm = llm
        self._lock = asyncio.Lock()  # модели не thread-safe -> сериализуем доступ

    @classmethod
    async def build(cls, cfg) -> "AssistantPipeline":
        loop = asyncio.get_running_loop()
        # Загрузка тяжёлых моделей в executor, чтобы не блокировать event loop.
        asr = await loop.run_in_executor(
            None,
            lambda: AsrEmotionPipeline.load(
                whisper_size=cfg.asr_model_size,
                compute_type=cfg.asr_compute_type,
                emotion_model_id=cfg.emotion_model,
                device=cfg.device,
            ),
        )
        llm = await loop.run_in_executor(
            None,
            lambda: GemmaPlannerLLM.load(
                model_path=cfg.llm_model_path,
                quant=cfg.llm_quant,
                device=cfg.device,
            ),
        )
        return cls(asr=asr, llm=llm)

    async def process_voice(self, audio_path: Path) -> PlannerTask:
        async with self._lock:
            loop = asyncio.get_running_loop()
            enriched: EnrichedUtterance = await loop.run_in_executor(
                None, self.asr.transcribe_with_emotion, str(audio_path)
            )
            logger.info("ASR+эмоция: %s", enriched.to_prompt())
            task: PlannerTask = await loop.run_in_executor(
                None, self.llm.to_task, enriched
            )
            return task

    async def process_text(self, text: str) -> PlannerTask:
        from shared.schemas import Emotion, EmotionResult
        enriched = EnrichedUtterance(
            text=text,
            emotion=EmotionResult(label=Emotion.NEUTRAL, score=1.0, is_urgent=False),
            source="text",
        )
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.llm.to_task, enriched)
