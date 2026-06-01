"""Форматирование задачи в читаемое сообщение Telegram."""
from __future__ import annotations

from datetime import date

from shared.deadline import resolve_deadline
from shared.schemas import PlannerTask

_PRIORITY_BADGE = {
    "urgent": "🔴 СРОЧНО",
    "high": "🟠 высокий",
    "medium": "🟡 средний",
    "low": "🟢 низкий",
}


def _fmt_deadline(phrase: str | None, today: date | None = None) -> str:
    """Показываем нормализованную фразу + конкретную дату, если резолвится."""
    if not phrase or phrase.strip().lower() in {"без срока", ""}:
        return "без срока"
    resolved = resolve_deadline(phrase, today or date.today())
    return f"{phrase} ({resolved})" if resolved else phrase


def format_task(task: PlannerTask, today: date | None = None) -> str:
    badge = _PRIORITY_BADGE.get(task.priority, task.priority)
    lines = [
        f"<b>{task.title}</b>",
        f"Приоритет: {badge}",
        f"Срок: {_fmt_deadline(task.deadline, today)}",
    ]
    if task.description:
        lines.append(f"\n{task.description}")

    if task.checkpoints:
        lines.append("\n<b>План:</b>")
        for i, cp in enumerate(task.checkpoints, 1):
            sub = ""
            if cp.deadline:
                sub = f" — <i>{_fmt_deadline(cp.deadline, today)}</i>"
            lines.append(f"{i}. {cp.step}{sub}")

    if task.tags:
        lines.append("\n" + " ".join(f"#{t}" for t in task.tags))

    return "\n".join(lines)
