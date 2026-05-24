"""Промпты для FunctionGemma — без эмоций, только текст → JSON-задача."""
from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "Ты — функция-планировщик. По входному сообщению пользователя "
    "(расшифрованному из голоса) ты обязана вернуть СТРОГО ОДИН JSON-объект, "
    "без лишнего текста, по схеме:\n"
    "{\n"
    '  "title": str,\n'
    '  "description": str | null,\n'
    '  "deadline": str | null,\n'
    '  "priority": "low" | "medium" | "high"\n'
    "}\n"
    "Никогда не пиши пояснений, только JSON."
)

FEW_SHOT = [
    {
        "user": "завтра в девять утра у меня собес, надо подготовить ответы",
        "assistant": json.dumps({
            "title": "Подготовиться к собеседованию",
            "description": "Подготовить ответы для собеседования завтра в 9:00",
            "deadline": "завтра в 09:00",
            "priority": "high",
        }, ensure_ascii=False),
    },
    {
        "user": "купить молоко по дороге домой",
        "assistant": json.dumps({
            "title": "Купить молоко",
            "description": "Заехать за молоком по дороге домой",
            "deadline": "без срока",
            "priority": "low",
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