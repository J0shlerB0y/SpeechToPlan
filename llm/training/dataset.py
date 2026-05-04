"""Подготовка датасета для дообучения FunctionGemma на задаче «текст→JSON-задача».

Формат входного JSONL:
    {"input": "[Эмоция: urgent] Текст: ...", "output": {"title": "...", ...}}

Файл tasks_train.jsonl и tasks_val.jsonl кладутся в llm/data/.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


def load_jsonl(path: str | Path) -> Dataset:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def to_chat_examples(ds: Dataset, tokenizer, system_prompt: str, max_len: int = 1024) -> Dataset:
    """Превращает {input, output} в токенизированные обучающие примеры."""

    def _format(example):
        target = json.dumps(example["output"], ensure_ascii=False)
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": target},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=max_len)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    return ds.map(_format, remove_columns=ds.column_names)
