"""Промпты и JSON-схема для FunctionGemma."""
from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "Ты — функция-планировщик. По входному сообщению пользователя "
    "(уже расшифрованному из голоса и помеченному эмоцией) ты обязана "
    "вернуть СТРОГО ОДИН JSON-объект, без лишнего текста, по схеме:\n"
    "{\n"
    '  "title": str,\n'
    '  "description": str | null,\n'
    '  "due_date": "YYYY-MM-DD" | null,\n'
    '  "due_time": "HH:MM" | null,\n'
    '  "priority": "low" | "medium" | "high" | "critical",\n'
    '  "tags": [str],\n'
    '  "raw_emotion": str | null\n'
    "}\n"
    "Если эмоция URGENT/anxious/angry — повышай priority минимум до high. "
    "Никогда не пиши пояснений, только JSON."
)

FEW_SHOT = [
    {
        "user": "[Эмоция: urgent | URGENT] Текст: завтра в девять утра у меня собес, надо подготовить ответы",
        "assistant": json.dumps({
            "title": "Подготовиться к собесу",
            "description": "Подготовить ответы для собеседования",
            "due_date": None, "due_time": "09:00",
            "priority": "high", "tags": ["карьера", "собеседование"],
            "raw_emotion": "urgent",
        }, ensure_ascii=False),
    },
    {
        "user": "[Эмоция: neutral] Текст: купить молоко по дороге домой",
        "assistant": json.dumps({
            "title": "Купить молоко",
            "description": "По дороге домой",
            "due_date": None, "due_time": None,
            "priority": "low", "tags": ["покупки"],
            "raw_emotion": "neutral",
        }, ensure_ascii=False),
    },
]


def build_chat(user_prompt: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs
