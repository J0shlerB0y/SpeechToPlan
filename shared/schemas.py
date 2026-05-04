"""Общие Pydantic-схемы, которыми обмениваются модули bot / asr_emotion / llm."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    URGENT = "urgent"


class ASRResult(BaseModel):
    text: str
    language: str = "ru"
    duration_sec: float = 0.0
    avg_logprob: float = 0.0


class EmotionResult(BaseModel):
    label: Emotion = Emotion.NEUTRAL
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    is_urgent: bool = False


class EnrichedUtterance(BaseModel):
    """Объединённый результат ASR + эмоций — то, что уходит в LLM."""
    text: str
    emotion: EmotionResult
    source: str = "voice"

    def to_prompt(self) -> str:
        tag = self.emotion.label.value
        urgency = " | URGENT" if self.emotion.is_urgent else ""
        return f"[Эмоция: {tag}{urgency}] Текст: {self.text}"


class PlannerTask(BaseModel):
    """Финальный JSON, который бот отдаёт пользователю / планировщику."""
    title: str
    description: Optional[str] = None
    due_date: Optional[str] = None  # ISO-8601 YYYY-MM-DD
    due_time: Optional[str] = None  # HH:MM
    priority: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    tags: list[str] = []
    raw_emotion: Optional[str] = None
