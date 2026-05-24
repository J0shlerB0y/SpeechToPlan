"""Дообучение FunctionGemma-270m-it через QLoRA + явный Grid Search.

Академическое требование: вместо HPO-фреймворков (Optuna/Ray) реализован
УЧЕБНЫЙ перебор по сетке (learning_rate × lora_rank × lora_alpha) — это даёт
прозрачную и воспроизводимую таблицу результатов для защиты.

Метрика выбора: minimum eval_loss на валидационной выборке.
Для каждой комбинации сохраняется отдельный adapter; в конце копируется
лучший в `<output>/best_adapter/`.

Запуск:
    python -m llm.training.qlora_grid_search \
        --train-file ./llm/data/tasks_train.jsonl \
        --val-file ./llm/data/tasks_val.jsonl \
        --output ./llm/checkpoints/gemma-grid
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm.prompts import SYSTEM_PROMPT
from llm.training.dataset import load_jsonl, to_chat_examples

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("qlora_grid")


@dataclass
class GridPoint:
    learning_rate: float
    lora_rank: int
    lora_alpha: int

    def slug(self) -> str:
        return f"lr{self.learning_rate:.0e}_r{self.lora_rank}_a{self.lora_alpha}"


@dataclass
class GridResult:
    point: GridPoint
    eval_loss: float
    train_loss: float
    adapter_dir: str


def _bnb_4bit() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _load_base(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=_bnb_4bit(),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return tokenizer, model


def _train_one(
    point: GridPoint,
    base_model_path: str,
    train_ds,
    eval_ds,
    tokenizer,
    output_root: Path,
    epochs: int,
    batch: int,
    grad_accum: int,
) -> GridResult:
    log.info("=== Grid point %s ===", point.slug())

    # Каждый запуск стартует со свежей копии базовой модели,
    # чтобы LoRA-адаптеры не наслаивались между точками сетки.
    _, model = _load_base(base_model_path)

    lora_cfg = LoraConfig(
        r=point.lora_rank,
        lora_alpha=point.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    run_dir = output_root / point.slug()
    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=point.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        fp16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    train_metrics = trainer.train().metrics
    eval_metrics = trainer.evaluate()

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Освобождаем VRAM перед следующей итерацией Grid Search.
    del trainer, model
    torch.cuda.empty_cache()

    return GridResult(
        point=point,
        eval_loss=float(eval_metrics["eval_loss"]),
        train_loss=float(train_metrics.get("train_loss", float("nan"))),
        adapter_dir=str(adapter_dir),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/functiongemma-270m-it")
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--output", default="./llm/checkpoints/gemma-grid")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lrs", nargs="+", type=float, default=[5e-5, 1e-4, 3e-4])
    p.add_argument("--ranks", nargs="+", type=int, default=[8, 16])
    p.add_argument("--alphas", nargs="+", type=int, default=[16, 32])
    args = p.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer, _ = _load_base(args.base_model)
    train_raw = load_jsonl(args.train_file)
    eval_raw = load_jsonl(args.val_file)
    train_ds = to_chat_examples(train_raw, tokenizer, SYSTEM_PROMPT)
    eval_ds = to_chat_examples(eval_raw, tokenizer, SYSTEM_PROMPT)

    # ------- Явный перебор по сетке -------
    grid = [
        GridPoint(learning_rate=lr, lora_rank=r, lora_alpha=a)
        for lr, r, a in itertools.product(args.lrs, args.ranks, args.alphas)
    ]
    log.info("Размер сетки: %d точек", len(grid))

    results: list[GridResult] = []
    for i, point in enumerate(grid, 1):
        log.info("[%d/%d] training %s", i, len(grid), point.slug())
        try:
            result = _train_one(
                point=point,
                base_model_path=args.base_model,
                train_ds=train_ds,
                eval_ds=eval_ds,
                tokenizer=tokenizer,
                output_root=output_root,
                epochs=args.epochs,
                batch=args.batch,
                grad_accum=args.grad_accum,
            )
        except torch.cuda.OutOfMemoryError:
            log.exception("OOM на точке %s — пропускаем", point.slug())
            torch.cuda.empty_cache()
            continue
        results.append(result)
        log.info("→ eval_loss=%.4f", result.eval_loss)

    if not results:
        raise RuntimeError("Ни одна точка сетки не отработала.")

    # Сводная таблица — пригодится в защите.
    summary_path = output_root / "grid_summary.json"
    summary_path.write_text(
        json.dumps([{**asdict(r.point), "eval_loss": r.eval_loss,
                     "train_loss": r.train_loss, "adapter": r.adapter_dir} for r in results],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Grid summary -> %s", summary_path)

    best = min(results, key=lambda r: r.eval_loss)
    log.info("Лучшая точка: %s | eval_loss=%.4f", best.point.slug(), best.eval_loss)

    best_dir = output_root / "best_adapter"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best.adapter_dir, best_dir)
    log.info("Лучший адаптер скопирован в %s", best_dir)


if __name__ == "__main__":
    main()
