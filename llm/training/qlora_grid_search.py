"""Дообучение Qwen2.5-1.5B-Instruct через QLoRA + явный Grid Search (двухфазно).

Академическое требование: вместо HPO-фреймворков (Optuna/Ray) реализован
УЧЕБНЫЙ перебор по сетке (learning_rate × lora_rank × lora_alpha) — это даёт
прозрачную и воспроизводимую таблицу результатов для защиты.

Двухфазная схема:
  * ФАЗА 1 — короткий grid (--epochs) для выбора лучшей точки (lr/rank/alpha);
  * ФАЗА 2 — длинное финальное обучение лучшей точки (--final-epochs).

Режимы квантования:
  * --quant nf4  → QLoRA (CUDA + bitsandbytes), Qwen1.5B ~3 ГБ при обучении;
  * --quant none → обычное LoRA в fp16/fp32 (CPU / без bitsandbytes).

Метрики на каждой точке (под структурную схему с checkpoints):
  * train_loss, eval_loss, eval_perplexity = exp(eval_loss)
  * json_validity         — доля ответов, прошедших json.loads
  * exact_match           — побитовое совпадение (при вложенности почти всегда 0)
  * priority_accuracy     — доля совпавших priority
  * deadline_accuracy     — доля совпавших deadline
  * checkpoint_presence   — доля ответов с непустым checkpoints
  * checkpoint_count_match— доля, где число шагов близко к эталону (±1)

Запуск (полный, GPU):
    python -m llm.training.qlora_grid_search \
        --quant nf4 --device cuda \
        --epochs 3 --final-epochs 20 \
        --lrs 1e-4 3e-4 --ranks 16 --alphas 32
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm.prompts import SYSTEM_PROMPT, build_chat
from llm.training.dataset import load_jsonl, to_chat_examples

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("qlora_grid")


# --------------------------------------------------------------------------- #
# Точка сетки и её результат
# --------------------------------------------------------------------------- #
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
    train_loss: float
    eval_loss: float
    eval_perplexity: float
    json_validity: float
    exact_match: float
    priority_accuracy: float
    deadline_accuracy: float
    checkpoint_presence: float
    checkpoint_count_match: float
    adapter_dir: str


# --------------------------------------------------------------------------- #
# Загрузка базовой модели — два режима
# --------------------------------------------------------------------------- #
def _bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _load_base(model_path: str, quant: str, device: str):
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"token": hf_token}
    if quant == "nf4":
        if device != "cuda":
            raise RuntimeError("quant=nf4 (QLoRA) требует CUDA-torch")
        kwargs.update(
            quantization_config=_bnb_config(),
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        if device == "cuda":
            kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if device != "cuda":
        model.to(device)

    # gradient checkpointing экономит память; для nf4 это обязательно через
    # prepare_model_for_kbit_training, для fp32/16 — просто .enable()
    if quant == "nf4":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    return tokenizer, model


# --------------------------------------------------------------------------- #
# Кастомные метрики после обучения
# --------------------------------------------------------------------------- #
class _StopOnBalancedJson:
    """Останавливаем generation, когда фигурные скобки сбалансировались.

    ВАЖНО: со вложенными checkpoints в JSON несколько '}' — нельзя стопить на
    первой. Декодируем накопленный хвост и считаем баланс '{' и '}': как только
    был хотя бы один '{' и баланс вернулся к 0 — весь объект завершён.
    """
    def __init__(self, tokenizer, prompt_len: int):
        self.tok = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0, self.prompt_len:]
        if gen.numel() == 0:
            return False
        text = self.tok.decode(gen, skip_special_tokens=True)
        opened = text.count("{")
        if opened == 0:
            return False
        return text.count("}") >= opened


def _generate_eval_samples(model, tokenizer, eval_raw, device, max_new_tokens: int = 448) -> list[dict]:
    """Прогоняет модель по eval-выборке, возвращает [{pred, gold}].

    Стоп по балансу скобок (поддерживает вложенные checkpoints).
    """
    from transformers import StoppingCriteriaList

    rows = []
    model.eval()
    model_device = next(model.parameters()).device

    for ex in eval_raw:
        messages = build_chat(ex["input"])
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        enc = {k: v.to(model_device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[-1]
        stop_list = StoppingCriteriaList([_StopOnBalancedJson(tokenizer, prompt_len)])
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stop_list,
                use_cache=True,
            )
        pred_text = tokenizer.decode(
            out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        rows.append({"pred": pred_text, "gold": ex["output"]})
    return rows


def _compute_extra_metrics(rows: list[dict]) -> dict:
    """Метрики под структурную схему (с checkpoints).

    EM при вложенности почти всегда 0 — оставлен как референс. Главные метрики:
    priority_accuracy, deadline_accuracy, checkpoint_presence (есть ли шаги),
    checkpoint_count_match (близко ли число шагов к эталону).
    """
    import re
    JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

    n = len(rows)
    valid = em = 0
    prio_hit = prio_tot = 0
    dl_hit = dl_tot = 0
    cp_present = 0           # сколько предсказаний содержат непустой checkpoints
    cp_count_match = 0       # |len(pred_cp) - len(gold_cp)| <= 1
    cp_total_gold = 0        # у скольких эталонов есть checkpoints

    for r in rows:
        gold = r["gold"]
        pred_text = r["pred"]
        m = JSON_RE.search(pred_text)
        if not m:
            continue
        try:
            pred = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        valid += 1

        if pred == gold:
            em += 1

        if "priority" in gold:
            prio_tot += 1
            if pred.get("priority") == gold.get("priority"):
                prio_hit += 1
        if "deadline" in gold:
            dl_tot += 1
            if pred.get("deadline") == gold.get("deadline"):
                dl_hit += 1

        gold_cp = gold.get("checkpoints") or []
        pred_cp = pred.get("checkpoints") or []
        if isinstance(pred_cp, list) and len(pred_cp) > 0:
            cp_present += 1
        if gold_cp:
            cp_total_gold += 1
            if isinstance(pred_cp, list) and abs(len(pred_cp) - len(gold_cp)) <= 1:
                cp_count_match += 1

    return {
        "json_validity": valid / n if n else 0.0,
        "exact_match": em / n if n else 0.0,
        "priority_accuracy": prio_hit / prio_tot if prio_tot else 0.0,
        "deadline_accuracy": dl_hit / dl_tot if dl_tot else 0.0,
        "checkpoint_presence": cp_present / n if n else 0.0,
        "checkpoint_count_match": cp_count_match / cp_total_gold if cp_total_gold else 0.0,
    }


# Обучение одной точки сетки
def _train_one(
    point: GridPoint,
    base_model_path: str,
    train_ds,
    eval_ds,
    eval_raw,
    tokenizer,
    output_root: Path,
    epochs: int,
    batch: int,
    grad_accum: int,
    quant: str,
    device: str,
) -> GridResult:
    log.info("=== Grid point %s (quant=%s) ===", point.slug(), quant)
    _, model = _load_base(base_model_path, quant=quant, device=device)

    lora_cfg = LoraConfig(
        r=point.lora_rank,
        lora_alpha=point.lora_alpha,
        # attention + MLP проекции — больше ёмкости для structured-вывода.
        # Имена модулей у Qwen2.5 совпадают с Llama/SmolLM2.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    run_dir = output_root / point.slug()
    optim = "paged_adamw_8bit" if quant == "nf4" else "adamw_torch"
    fp16 = device == "cuda"

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
        optim=optim,
        fp16=fp16,
        logging_steps=5,
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
        processing_class=tokenizer,
    )

    train_metrics = trainer.train().metrics
    eval_metrics = trainer.evaluate()
    eval_loss = float(eval_metrics["eval_loss"])

    # Extra metrics: actually generate and compare to gold.
    # Ограничиваем max 20 примерами чтобы не зависнуть (~2-3 мин вместо 10+).
    eval_subset = eval_raw[:20]
    log.info("Считаем дополнительные метрики (generate × %d из %d)", len(eval_subset), len(eval_raw))
    rows = _generate_eval_samples(model, tokenizer, eval_subset, device=device, max_new_tokens=256)
    extra = _compute_extra_metrics(rows)

    # Сохраняем pred/gold для прозрачности (приложим к защите)
    (run_dir / "predictions.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    del trainer, model
    if device == "cuda":
        torch.cuda.empty_cache()

    return GridResult(
        point=point,
        train_loss=float(train_metrics.get("train_loss", float("nan"))),
        eval_loss=eval_loss,
        eval_perplexity=math.exp(min(eval_loss, 20.0)),  # cap чтоб не было inf
        json_validity=extra["json_validity"],
        exact_match=extra["exact_match"],
        priority_accuracy=extra["priority_accuracy"],
        deadline_accuracy=extra["deadline_accuracy"],
        checkpoint_presence=extra["checkpoint_presence"],
        checkpoint_count_match=extra["checkpoint_count_match"],
        adapter_dir=str(adapter_dir),
    )


# Main: Grid Search
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-file", default="./llm/data/tasks_train_v2.jsonl")
    p.add_argument("--val-file", default="./llm/data/tasks_val_v2.jsonl")
    p.add_argument("--output", default="./checkpoints/qwen-grid")
    p.add_argument("--quant", default="nf4", choices=["nf4", "none"],
                   help="nf4 — QLoRA (нужна CUDA+bitsandbytes); none — обычное LoRA fp16/fp32")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=3,
                   help="эпох в ФАЗЕ 1 (grid search для выбора lr/rank)")
    p.add_argument("--final-epochs", type=int, default=20,
                   help="эпох в ФАЗЕ 2 (длинное финальное обучение лучшей точки); 0 = пропустить")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4])
    p.add_argument("--ranks", nargs="+", type=int, default=[16])
    p.add_argument("--alphas", nargs="+", type=int, default=[32])
    p.add_argument("--train-limit", type=int, default=0,
                   help="0 = весь train; иначе подсэмплировать N примеров (для CPU)")
    p.add_argument("--val-limit", type=int, default=0,
                   help="0 = весь val; иначе подсэмплировать N примеров")
    args = p.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    log.info("Базовая модель: %s | device=%s | quant=%s",
             args.base_model, args.device, args.quant)

    tokenizer, _ = _load_base(args.base_model, quant=args.quant, device=args.device)
    train_raw = load_jsonl(args.train_file)
    val_raw   = load_jsonl(args.val_file)

    if args.train_limit and len(train_raw) > args.train_limit:
        train_raw = train_raw.select(range(args.train_limit))
    if args.val_limit and len(val_raw) > args.val_limit:
        val_raw = val_raw.select(range(args.val_limit))

    eval_raw_list = list(val_raw)
    train_ds = to_chat_examples(train_raw, tokenizer, SYSTEM_PROMPT)
    eval_ds  = to_chat_examples(val_raw,   tokenizer, SYSTEM_PROMPT)

    log.info("Train: %d примеров | Val: %d примеров", len(train_ds), len(eval_ds))

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
                eval_raw=eval_raw_list,
                tokenizer=tokenizer,
                output_root=output_root,
                epochs=args.epochs,
                batch=args.batch,
                grad_accum=args.grad_accum,
                quant=args.quant,
                device=args.device,
            )
        except torch.cuda.OutOfMemoryError:
            log.exception("OOM на точке %s — пропускаем", point.slug())
            torch.cuda.empty_cache()
            continue
        results.append(result)
        log.info(
            "→ eval_loss=%.4f ppl=%.2f valid=%.0f%% prio=%.0f%% deadline=%.0f%% cp=%.0f%%",
            result.eval_loss, result.eval_perplexity,
            100 * result.json_validity, 100 * result.priority_accuracy,
            100 * result.deadline_accuracy, 100 * result.checkpoint_presence,
        )

    if not results:
        raise RuntimeError("Ни одна точка сетки не отработала.")

    def _summarize(rs: list[GridResult]) -> list[dict]:
        return [
            {
                **asdict(r.point),
                "train_loss": r.train_loss,
                "eval_loss": r.eval_loss,
                "eval_perplexity": r.eval_perplexity,
                "json_validity": r.json_validity,
                "exact_match": r.exact_match,
                "priority_accuracy": r.priority_accuracy,
                "deadline_accuracy": r.deadline_accuracy,
                "checkpoint_presence": r.checkpoint_presence,
                "checkpoint_count_match": r.checkpoint_count_match,
                "adapter": r.adapter_dir,
            }
            for r in rs
        ]

    (output_root / "grid_summary.json").write_text(
        json.dumps(_summarize(results), ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Grid summary -> %s", output_root / "grid_summary.json")

    # Композитный критерий: качество полей (priority+deadline+checkpoints), потом -eval_loss
    def _score(r: GridResult) -> tuple:
        task_q = r.priority_accuracy + r.deadline_accuracy + r.checkpoint_count_match
        return (task_q, -r.eval_loss)

    best = max(results, key=_score)
    log.info("Лучшая точка (Фаза 1): %s | eval_loss=%.4f prio=%.2f deadline=%.2f",
             best.point.slug(), best.eval_loss, best.priority_accuracy, best.deadline_accuracy)

    final_result = best
    if args.final_epochs and args.final_epochs > args.epochs:
        log.info("=== Фаза 2: финальное обучение %s на %d эпох ===",
                 best.point.slug(), args.final_epochs)
        final_result = _train_one(
            point=best.point,
            base_model_path=args.base_model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            eval_raw=eval_raw_list,
            tokenizer=tokenizer,
            output_root=output_root / "final",
            epochs=args.final_epochs,
            batch=args.batch,
            grad_accum=args.grad_accum,
            quant=args.quant,
            device=args.device,
        )
        (output_root / "final_summary.json").write_text(
            json.dumps(_summarize([final_result]), ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Финал: eval_loss=%.4f prio=%.2f deadline=%.2f cp_count=%.2f",
                 final_result.eval_loss, final_result.priority_accuracy,
                 final_result.deadline_accuracy, final_result.checkpoint_count_match)

    best_dir = output_root / "best_adapter"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(final_result.adapter_dir, best_dir)
    log.info("Лучший адаптер скопирован в %s", best_dir)


if __name__ == "__main__":
    main()
