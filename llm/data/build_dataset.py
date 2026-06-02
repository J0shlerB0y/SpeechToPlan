"""Сборка датасета v2 (гибрид): обогащение существующих 434 + ручной seed.

Что делает:
1. Грузит tasks_train.jsonl / tasks_val.jsonl (старый плоский формат).
2. ОБОГАЩАЕТ каждый пример:
   - priority по правилам (ключевые слова срочности/важности + близость дедлайна),
   - нормализует deadline к каноническому словарю,
   - генерирует 2-4 контрольные точки (checkpoints) по категории задачи.
3. Подмешивает ручной complex_seed.jsonl (сложные многошаговые кейсы с urgent).
4. Пишет tasks_train_v2.jsonl / tasks_val_v2.jsonl.

Запуск:
    python -m llm.data.build_dataset
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)
DATA_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Правила приоритета
# --------------------------------------------------------------------------- #
URGENT_WORDS = [
    "срочно", "срочн", "немедленно", "горит", "кровь из носу", "аврал",
    "пожар", "крайний срок сегодня", "вот-вот", "asap", "очень срочно",
]
HIGH_WORDS = [
    "важно", "важн", "не забыть", "обязательно", "штраф", "экзамен", "собес",
    "собеседование", "дедлайн", "критично", "нельзя пропустить", "ответствен",
    "налог", "оплат", "счета", "врач", "виза", "паспорт", "отчёт", "отчет",
]
LOW_WORDS = [
    "когда-нибудь", "не к спеху", "на досуге", "без срока", "при случае",
    "если будет время", "не срочно", "потом", "как-нибудь",
]


def infer_priority(text: str, current: str, deadline: str) -> str:
    t = text.lower()
    dl = (deadline or "").lower()
    if any(w in t for w in URGENT_WORDS):
        return "urgent"
    if any(w in t for w in LOW_WORDS):
        return "low"
    if any(w in t for w in HIGH_WORDS):
        return "high"
    # bump по близости срока
    if "сегодня" in dl:
        return "high"
    if "завтра" in dl and current == "low":
        return "medium"
    return current or "medium"


# --------------------------------------------------------------------------- #
# Нормализация дедлайнов
# --------------------------------------------------------------------------- #
DEADLINE_MAP = {
    "пятница": "в пятницу",
    "понедельник": "в понедельник",
    "вторник": "во вторник",
    "среда": "в среду",
    "четверг": "в четверг",
    "суббота": "в субботу",
    "воскресенье": "в воскресенье",
    "выходные": "на выходных",
    "до 15 числа": "до 15-го числа",
    "до 20 числа": "до 20-го числа",
    "не указан": "без срока",
    "": "без срока",
}


def normalize_deadline(deadline: str | None) -> str:
    if not deadline:
        return "без срока"
    d = deadline.strip()
    return DEADLINE_MAP.get(d.lower(), d)


# --------------------------------------------------------------------------- #
# Категории задач → шаблоны контрольных точек
# --------------------------------------------------------------------------- #
# Каждая категория: (ключевые слова, функция-шаблон шагов).
# Шаги — осмысленная декомпозиция; последний наследует общий дедлайн.
CATEGORY_RULES: list[tuple[list[str], list[str]]] = [
    (["встреч", "собес", "переговор", "клиент", "созвон", "совещан"],
     ["Уточнить время и место", "Подготовить материалы и вопросы", "Провести встречу"]),
    (["оплат", "счета", "счёт", "штраф", "налог", "коммунал", "платёж", "платеж"],
     ["Проверить сумму и реквизиты", "Произвести оплату", "Сохранить квитанцию"]),
    (["купить", "покупка", "приобрести", "заказать"],
     ["Составить список", "Сравнить цены", "Совершить покупку"]),
    (["экзамен", "зачёт", "зачет", "тест", "контрольн", "подготов", "учить", "лабораторн", "курсов"],
     ["Собрать материалы и конспекты", "Проработать сложные темы", "Повторить перед сдачей"]),
    (["врач", "приём", "прием", "анализ", "обследован", "медиц", "стоматолог", "поликлин"],
     ["Записаться на приём", "Собрать документы и анализы", "Посетить врача"]),
    (["документ", "виза", "паспорт", "справк", "заявлен", "оформ"],
     ["Уточнить список документов", "Собрать и заполнить", "Подать на оформление"]),
    (["тренировк", "марафон", "спорт", "бег", "зал", "фитнес"],
     ["Составить план тренировок", "Подготовить экипировку", "Провести тренировку"]),
    (["позвонить", "звонок", "связаться", "написать", "сообщить"],
     ["Найти контакт", "Подготовить, что сказать", "Совершить звонок"]),
    (["презентац", "доклад", "отчёт", "отчет", "проект", "защит"],
     ["Собрать данные и структуру", "Подготовить черновик", "Финализировать и проверить"]),
    (["поездк", "путешеств", "билет", "бронир", "отель", "отпуск"],
     ["Выбрать даты и маршрут", "Забронировать билеты и жильё", "Собрать вещи"]),
    (["ремонт", "почин", "устран", "настро"],
     ["Определить проблему", "Подготовить инструменты/детали", "Выполнить работу"]),
]
DEFAULT_STEPS = ["Уточнить детали", "Выполнить основную часть", "Проверить результат"]


def make_checkpoints(text: str, deadline: str) -> list[dict]:
    t = text.lower()
    steps = DEFAULT_STEPS
    for keywords, tmpl in CATEGORY_RULES:
        if any(k in t for k in keywords):
            steps = tmpl
            break
    cps = [{"step": s, "deadline": None} for s in steps]
    # Последний шаг получает общий дедлайн (если он конкретный)
    if deadline and deadline != "без срока":
        cps[-1]["deadline"] = deadline
    return cps


# --------------------------------------------------------------------------- #
# Обогащение одного примера
# --------------------------------------------------------------------------- #
def enrich(example: dict) -> dict:
    inp = example["input"]
    out = dict(example["output"])  # копия

    deadline = normalize_deadline(out.get("deadline"))
    priority = infer_priority(inp, out.get("priority", "medium"), deadline)
    checkpoints = make_checkpoints(inp, deadline)

    new_out = {
        "title": out.get("title") or inp[:50],
        "description": out.get("description") or inp,
        "deadline": deadline,
        "priority": priority,
        "checkpoints": checkpoints,
    }
    return {"input": inp, "output": new_out}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    train_raw = load_jsonl(DATA_DIR / "tasks_train.jsonl")
    val_raw = load_jsonl(DATA_DIR / "tasks_val.jsonl")

    train = [enrich(e) for e in train_raw]
    val = [enrich(e) for e in val_raw]

    # Ручной seed сложных кейсов
    seed_path = DATA_DIR / "complex_seed.jsonl"
    seed = load_jsonl(seed_path) if seed_path.exists() else []
    print(f"Seed сложных кейсов: {len(seed)}")

    # Делим seed: 85% в train, 15% в val
    random.shuffle(seed)
    n_val = max(1, int(len(seed) * 0.15)) if seed else 0
    seed_val = seed[:n_val]
    seed_train = seed[n_val:]

    train.extend(seed_train)
    val.extend(seed_val)
    random.shuffle(train)
    random.shuffle(val)

    write_jsonl(DATA_DIR / "tasks_train_v2.jsonl", train)
    write_jsonl(DATA_DIR / "tasks_val_v2.jsonl", val)

    # Статистика
    from collections import Counter
    prio = Counter(r["output"]["priority"] for r in train)
    avg_cp = sum(len(r["output"].get("checkpoints", [])) for r in train) / max(1, len(train))
    print(f"Train v2: {len(train)} | Val v2: {len(val)}")
    print(f"Priority распределение (train): {dict(prio)}")
    print(f"Среднее число checkpoints (train): {avg_cp:.2f}")


if __name__ == "__main__":
    main()
