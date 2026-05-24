"""Подготовка датасета для дообучения FunctionGemma на задаче «текст→JSON-задача».

Формат входного JSONL:
    {"input": "[Эмоция: urgent] Текст: ...", "output": {"title": "...", ...}}

Файл tasks_train.jsonl и tasks_val.jsonl кладутся в llm/data/.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value


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

    def _format(example):
        target = json.dumps(example["output"], ensure_ascii=False)
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": target},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length")

        pad_id = tokenizer.pad_token_id
        labels = [
            token_id if token_id != pad_id else -100
            for token_id in enc["input_ids"]
        ]

        return {
            "input_ids": list(enc["input_ids"]),
            "attention_mask": list(enc["attention_mask"]),
            "labels": labels,
        }

    features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })

    return ds.map(_format, remove_columns=ds.column_names, features=features)