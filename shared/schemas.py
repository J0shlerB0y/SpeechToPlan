"""Общие Pydantic-схемы, которыми обмениваются модули bot / asr_emotion / llm."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ASRResult(BaseModel):
    text: str
    language: str = "ru"
    duration_sec: float = 0.0
    avg_logprob: float = 0.0


class EnrichedUtterance(BaseModel):
    """Результат ASR — то, что уходит в LLM."""
    text: str
    source: str = "voice"

    def to_prompt(self) -> str:
        return self.text


class PlannerTask(BaseModel):
    """Финальный JSON, который бот отдаёт пользователю / планировщику."""
    title: str
    description: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = Field(default="medium", pattern="^(low|medium|high)$")