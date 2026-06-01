"""Резолвер дедлайнов: нормализованная русская фраза → конкретная дата.

Модель выдаёт срок в каноническом виде («завтра», «в пятницу», «через неделю»…),
а реальную дату вычисляет рантайм от текущего дня. Так модели не нужно знать,
какое сегодня число.

Пример:
    >>> from datetime import date
    >>> resolve_deadline("завтра", date(2026, 6, 1))
    '2026-06-02'
    >>> resolve_deadline("в пятницу", date(2026, 6, 1))   # пн → ближайшая пт
    '2026-06-05'
    >>> resolve_deadline("без срока", date(2026, 6, 1))
    None
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

# Канонический словарь дней недели (0 = понедельник).
_WEEKDAYS = {
    "понедельник": 0, "вторник": 1, "среда": 2, "среду": 2, "четверг": 3,
    "пятница": 4, "пятницу": 4, "суббота": 5, "субботу": 5,
    "воскресенье": 6, "воскресенья": 6,
}

# Фразы, означающие отсутствие конкретного срока.
_NO_DEADLINE = {"без срока", "не указан", "не указано", "никогда", "когда-нибудь", ""}


def _next_weekday(today: date, target_wd: int, allow_today: bool = False) -> date:
    """Ближайший день недели target_wd начиная с today."""
    delta = (target_wd - today.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return today + timedelta(days=delta)


def _extract_time(phrase: str) -> Optional[str]:
    """Достаёт HH:MM из фразы вида 'завтра 18:00' / 'до 9 утра'."""
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", phrase)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    m = re.search(r"\b(\d{1,2})\s*час", phrase)
    if m:
        return f"{int(m.group(1)):02d}:00"
    return None


def resolve_deadline(phrase: Optional[str], today: Optional[date] = None) -> Optional[str]:
    """Нормализованная фраза → 'YYYY-MM-DD' или 'YYYY-MM-DD HH:MM'.

    Возвращает None, если фраза пустая / без срока / не распознана как дата
    (в этом случае бот покажет исходную фразу как есть).
    """
    if not phrase:
        return None
    today = today or date.today()
    p = phrase.strip().lower()

    if p in _NO_DEADLINE:
        return None

    time_part = _extract_time(p)

    def _fmt(d: date) -> str:
        return f"{d.isoformat()} {time_part}" if time_part else d.isoformat()

    # Относительные якоря
    if "послезавтра" in p:
        return _fmt(today + timedelta(days=2))
    if "сегодня" in p or "до конца дня" in p:
        return _fmt(today)
    if "завтра" in p:
        return _fmt(today + timedelta(days=1))

    # «через N дней/недель/месяцев»
    m = re.search(r"через\s+(\d+|один|одну|два|две|три|четыре|пять)\s*(дн|недел|месяц)", p)
    if m:
        words = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5}
        n_raw = m.group(1)
        n = int(n_raw) if n_raw.isdigit() else words.get(n_raw, 1)
        unit = m.group(2)
        if unit.startswith("дн"):
            return _fmt(today + timedelta(days=n))
        if unit.startswith("недел"):
            return _fmt(today + timedelta(weeks=n))
        if unit.startswith("месяц"):
            return _fmt(today + timedelta(days=30 * n))
    if "через неделю" in p:
        return _fmt(today + timedelta(weeks=1))
    if "через месяц" in p:
        return _fmt(today + timedelta(days=30))

    if "на следующей неделе" in p:
        return _fmt(_next_weekday(today, 0))  # ближайший понедельник
    if "на этой неделе" in p:
        return _fmt(_next_weekday(today, 4, allow_today=True))  # до пятницы
    if "на выходных" in p or "выходны" in p:
        return _fmt(_next_weekday(today, 5))  # суббота

    # День недели
    for name, wd in _WEEKDAYS.items():
        if name in p:
            return _fmt(_next_weekday(today, wd))

    # «до N числа»
    m = re.search(r"до\s+(\d{1,2})(?:[\-\s]*го)?\s*числа", p)
    if m:
        day = int(m.group(1))
        year, month = today.year, today.month
        if day < today.day:  # число уже прошло → следующий месяц
            month += 1
            if month > 12:
                month, year = 1, year + 1
        try:
            return _fmt(date(year, month, day))
        except ValueError:
            return None

    # Не распознали — отдаём как есть (None → бот покажет исходную фразу)
    return None
