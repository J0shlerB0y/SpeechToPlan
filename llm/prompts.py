"""Промпты для LLM-планировщика (Qwen2.5-Instruct).

Qwen2.5 использует ChatML и НАТИВНО поддерживает role="system" — поэтому
системную инструкцию кладём отдельным system-сообщением (чище, чем инъекция
в первое user-сообщение, которая была нужна для Gemma).
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "Ты — функция-планировщик. По сообщению пользователя (часто расшифрованному "
    "из голоса) верни СТРОГО ОДИН JSON-объект, без пояснений и текста вокруг.\n\n"
    "Схема:\n"
    "{\n"
    '  "title": str,                       // короткое название задачи\n'
    '  "description": str,                 // 1-2 предложения, что и зачем\n'
    '  "deadline": str,                    // нормализованный срок или "без срока"\n'
    '  "priority": "low"|"medium"|"high"|"urgent",\n'
    '  "checkpoints": [                    // 2-4 осмысленных шага плана\n'
    '    {"step": str, "deadline": str|null}\n'
    "  ]\n"
    "}\n\n"
    "ПРАВИЛА ПРИОРИТЕТА:\n"
    "- urgent: есть слова «срочно/немедленно/горит/аврал» или срок «сегодня».\n"
    "- high: «важно/обязательно/не забыть», штрафы, экзамены, собеседования, "
    "оплаты, врач, документы.\n"
    "- low: «когда-нибудь/не к спеху/на досуге» или нет срока и низкая важность.\n"
    "- иначе medium.\n\n"
    "ПРАВИЛА СРОКА (deadline): нормализуй к фразам вида «сегодня», «завтра», "
    "«завтра 18:00», «в пятницу», «через неделю», «через две недели», «через месяц», "
    "«на выходных», «до 15-го числа». Если срока нет — «без срока».\n\n"
    "ПРАВИЛА ПЛАНА (checkpoints): разбей задачу на 2-4 последовательных шага. "
    "Каждый шаг — конкретное действие. Под-срок шага ставь, если он логичен, иначе null. "
    "Последний шаг обычно соответствует общему сроку задачи.\n"
)

FEW_SHOT = [
    {
        "user": "Срочно: завтра до 18:00 сдать отчёт шефу, ещё ничего не готово",
        "assistant": {
            "title": "Сдать отчёт шефу",
            "description": "Подготовить и сдать отчёт руководителю к завтрашнему вечеру",
            "deadline": "завтра 18:00",
            "priority": "urgent",
            "checkpoints": [
                {"step": "Собрать данные для отчёта", "deadline": "сегодня"},
                {"step": "Свести и оформить отчёт", "deadline": "завтра"},
                {"step": "Отправить руководителю", "deadline": "завтра 18:00"},
            ],
        },
    },
    {
        "user": "надо записаться к врачу на следующей неделе",
        "assistant": {
            "title": "Запись к врачу",
            "description": "Записаться и сходить на приём к врачу",
            "deadline": "на следующей неделе",
            "priority": "high",
            "checkpoints": [
                {"step": "Выбрать клинику и врача", "deadline": None},
                {"step": "Записаться на приём", "deadline": None},
                {"step": "Сходить на приём", "deadline": "на следующей неделе"},
            ],
        },
    },
    {
        "user": "когда-нибудь купить новую книгу по программированию",
        "assistant": {
            "title": "Купить книгу",
            "description": "Выбрать и купить книгу по программированию",
            "deadline": "без срока",
            "priority": "low",
            "checkpoints": [
                {"step": "Выбрать книгу по отзывам", "deadline": None},
                {"step": "Купить в магазине или онлайн", "deadline": None},
            ],
        },
    },
]


def _dump(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def build_chat(user_prompt: str, few_shot: bool = False) -> list[dict]:
    """Сообщения для инференса: system + (опц. few-shot) + текущий запрос.

    По умолчанию few_shot=False — это совпадает с форматом обучения
    (build_training_chat). Для дообученной модели few-shot не нужен: она уже
    выучила формат на 572 примерах, а лишний префикс только замедляет и создаёт
    рассинхрон train/inference.

    few_shot=True полезен для baseline-замера НЕдообученной базовой модели —
    там few-shot заметно помогает.
    """
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for ex in FEW_SHOT:
            msgs.append({"role": "user", "content": ex["user"]})
            msgs.append({"role": "assistant", "content": _dump(ex["assistant"])})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def build_training_chat(user_prompt: str, target_json: str) -> list[dict]:
    """Для обучения: system + текущий запрос + эталонный ответ.

    Few-shot в обучении НЕ кладём — модель учится отвечать на «голый» запрос
    под управлением system-инструкции. Это держит длину примера небольшой.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": target_json},
    ]
