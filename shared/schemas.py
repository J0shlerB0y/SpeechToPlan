from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ASRResult(BaseModel):
    text: str
    language: str = "ru"
    duration_sec: float = 0.0
    avg_logprob: float = 0.0


class EnrichedUtterance(BaseModel):
    """Результат ASR"""
    text: str
    source: str = "voice"

    def to_prompt(self) -> str:
        return self.text


class Checkpoint(BaseModel):
    """Одна контрольная точка плана."""
    step: str                          # что конкретно сделать
    deadline: Optional[str] = None     # под-срок шага (нормализованная фраза) или null


class PlannerTask(BaseModel):
    """Финальный JSON, который бот отдаёт пользователю."""
    title: str
    description: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = Field(default="medium", pattern="^(low|medium|high|urgent)$")
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
