"""Утилиты форматирования ответа пользователю."""
from __future__ import annotations

import json

from shared.schemas import PlannerTask


def format_task(task: PlannerTask) -> str:
    body = json.dumps(task.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
    return (
        f"*Задача добавлена* — `{task.priority.upper()}`\n"
        f"```json\n{body}\n```"
    )
