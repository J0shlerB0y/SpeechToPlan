"""Подготовка датасета для дообучения LLM на задаче «текст→JSON-задача».

Формат входного JSONL:
    {"input": "...текст...", "output": {"title": "...", "deadline": "...", ...}}

Файлы tasks_train.jsonl и tasks_val.jsonl лежат в llm/data/.

ВАЖНО: токенизация использует ту же функцию `build_training_chat`,
что и `llm/prompts.build_chat` (без role=system) — это гарантирует,
что модель видит одинаковый формат и при обучении, и при инференсе.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value

from llm.prompts import build_training_chat


def load_jsonl(path: str | Path) -> Dataset:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def to_chat_examples(
    ds: Dataset,
    tokenizer,
    system_prompt: str | None = None,  # совместимость со старой сигнатурой
    max_len: int = 768,  # макс. реальная длина примеров v2 ≈ 645 токенов
) -> Dataset:
    """Превращает {input, output} в токенизированные пары input_ids/labels."""

    def _format(example):
        target = json.dumps(example["output"], ensure_ascii=False)
        chat = build_training_chat(example["input"], target)
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False,
        )
        enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length")

        pad_id = tokenizer.pad_token_id
        labels = [
            tok if tok != pad_id else -100
            for tok in enc["input_ids"]
        ]
        return {
            "input_ids": list(enc["input_ids"]),
            "attention_mask": list(enc["attention_mask"]),
            "labels": labels,
        }

    features = Features({
        "input_ids":      Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels":         Sequence(Value("int64")),
    })
    return ds.map(_format, remove_columns=ds.column_names, features=features)
