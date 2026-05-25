from __future__ import annotations
import io
import av
import numpy as np
import soundfile
import argparse
import itertools
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, concatenate_datasets, load_dataset, Dataset, load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

load_dotenv()
print("HF_HOME:", os.getenv("HF_HOME"))
print("HF_DATASETS_CACHE:", os.getenv("HF_DATASETS_CACHE"))

# Проверяем что datasets тоже подхватил
from datasets import config as ds_config
print("datasets кэш:", ds_config.HF_DATASETS_CACHE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("train_whisper_lora")

HF_TOKEN = os.getenv("HF_TOKEN")

DATASETS = [
    {
        "repo":       "bond005/sberdevices_golos_10h_crowd",
        "config":     None,
        "text_field": "transcription",
        "max_train":  10_000,
        "max_eval":   500,
    },
    {
        "repo":       "bond005/taiga_speech_v2",
        "config":     None,
        "text_field": "transcription",
        "max_train":  12_000,
        "max_eval":   500,
    }
]

@dataclass
class GridPoint:
    lr: float
    lora_rank: int
    lora_alpha: int

    def slug(self) -> str:
        return f"lr{self.lr:.0e}_r{self.lora_rank}_a{self.lora_alpha}"


@dataclass
class GridResult:
    point: GridPoint
    wer: float
    cer: float
    adapter_dir: str


def _load_one(cfg: dict, split: str) -> Any | None:
    max_n = cfg["max_train"] if split == "train" else cfg["max_eval"]
    repo  = cfg["repo"]
    try:
        actual_split = split
        if split == "validation":
            for candidate in ["validation", "test"]:
                try:
                    load_dataset(repo, split=candidate, token=HF_TOKEN, streaming=True)
                    actual_split = candidate
                    break
                except Exception:
                    continue
            else:
                actual_split = "train"

        load_kwargs = dict(split=actual_split, token=HF_TOKEN, streaming=True)
        if cfg["config"]:
            ds = load_dataset(repo, cfg["config"], **load_kwargs)
        else:
            ds = load_dataset(repo, **load_kwargs)

        if split == "validation" and actual_split == "train":
            ds = ds.skip(cfg["max_train"])

        ds = Dataset.from_list(list(ds.take(max_n)))

        if cfg["text_field"] != "sentence":
            ds = ds.rename_column(cfg["text_field"], "sentence")

        keep = {"audio", "sentence"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        log.info("  ✓ %s [%s→%s]: %d примеров", repo, split, actual_split, len(ds))
        return ds

    except Exception as exc:
        log.warning("  ✗ %s [%s] пропущен: %s", repo, split, exc)
        return None


CACHE_DIR = Path("E:/Dev/Models/datasets/prepaired")  # куда сохраняем

def build_datasets() -> tuple[Any, Any]:
    cache_train = CACHE_DIR / "train"
    cache_eval  = CACHE_DIR / "eval"

    if cache_train.exists() and cache_eval.exists():
        log.info("Загружаем датасеты из кэша")
        return load_from_disk(str(cache_train)), load_from_disk(str(cache_eval))

    log.info("Первый запуск")
    trains, evals = [], []
    for cfg in DATASETS:
        tr = _load_one(cfg, "train")
        ev = _load_one(cfg, "validation")
        if tr: trains.append(tr)
        if ev: evals.append(ev)

    if not trains:
        raise RuntimeError("Ни один датасет не загрузился")

    train_ds = concatenate_datasets(trains).shuffle(seed=42)
    eval_ds  = concatenate_datasets(evals).shuffle(seed=42) if evals else \
               train_ds.select(range(min(500, len(train_ds))))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(cache_train))
    eval_ds.save_to_disk(str(cache_eval))
    log.info("Датасеты сохранены в %s", CACHE_DIR)

    return train_ds, eval_ds


def make_prepare_fn(processor: Any):
    def _prepare(batch):
        audio_info = batch["audio"]

        if audio_info.get("bytes"):
            container = av.open(io.BytesIO(audio_info["bytes"]))
        elif audio_info.get("path"):
            container = av.open(audio_info["path"])
        else:
            raise ValueError("Нет ни bytes ни path в audio")

        stream = container.streams.audio[0]
        samples = []
        for frame in container.decode(stream):
            samples.append(frame.to_ndarray().flatten())
        array = np.concatenate(samples).astype(np.float32)

        sr = stream.sample_rate
        if sr != 16_000:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=16_000)

        batch["input_features"] = processor.feature_extractor(
            array, sampling_rate=16_000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch
    return _prepare

@dataclass
class DataCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def make_metrics_fn(processor: Any):
    import evaluate
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def _compute(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        for p, r in zip(pred_str[:3], label_str[:3]):
            log.info("  pred : %s", p)
            log.info("  ref  : %s", r)

        return {"wer": wer, "cer": cer}

    return _compute


def load_model_with_lora(base_model: str, lora_rank: int, lora_alpha: int) -> Any:
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def train_one(
    point: GridPoint,
    base_model: str,
    processor: Any,
    train_ds: Any,
    eval_ds: Any,
    output_dir: str,
    epochs: int,
    batch: int,
    grad_accum: int,
) -> GridResult:
    log.info("--- %s ---", point.slug())

    model = load_model_with_lora(base_model, point.lora_rank, point.lora_alpha)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=point.lr,
        warmup_steps=50,
        num_train_epochs=epochs,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollator(processor=processor),
        processing_class=processor.feature_extractor,
        compute_metrics=make_metrics_fn(processor),
    )

    trainer.train()
    metrics = trainer.evaluate()

    wer = float(metrics.get("eval_wer", float("nan")))
    cer = float(metrics.get("eval_cer", float("nan")))
    log.info("%s → WER=%.2f%%  CER=%.2f%%", point.slug(), wer, cer)

    adapter_dir = Path(output_dir) / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    del trainer, model
    torch.cuda.empty_cache()

    return GridResult(point=point, wer=wer, cer=cer, adapter_dir=str(adapter_dir))


def main() -> None:
    p = argparse.ArgumentParser(description="Whisper LoRA fine-tune на русском")
    p.add_argument("--base-model",  default="openai/whisper-tiny")
    p.add_argument("--output",      default="./checkpoints/whisper-lora-ru")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch",       type=int,   default=4)
    p.add_argument("--grad-accum",  type=int,   default=8)
    # Сетка гиперпараметров
    p.add_argument("--lrs",         nargs="+",  type=float, default=[1e-3, 5e-4])
    p.add_argument("--ranks",       nargs="+",  type=int,   default=[8, 16])
    p.add_argument("--alphas",      nargs="+",  type=int,   default=[16])
    # Фаза 1: быстрый поиск на подвыборке
    p.add_argument("--search-samples", type=int, default=5_000,
                   help="Кол-во примеров для grid search (фаза 1)")
    p.add_argument("--search-epochs",  type=int, default=1,
                   help="Эпох для grid search (фаза 1)")
    args = p.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print(HF_TOKEN)

    processor = WhisperProcessor.from_pretrained(
        args.base_model, language="ru", task="transcribe", token=HF_TOKEN
    )

    train_raw, eval_raw = build_datasets()
    prepare_fn = make_prepare_fn(processor)
    prep_kwargs = dict(remove_columns=train_raw.column_names, num_proc=2)

    log.info("Подготовка train...")
    train_ds = train_raw.map(prepare_fn, **prep_kwargs)
    log.info("Подготовка eval...")
    eval_ds = eval_raw.map(prepare_fn, **{**prep_kwargs, "remove_columns": eval_raw.column_names})

    grid = [
        GridPoint(lr=lr, lora_rank=r, lora_alpha=a)
        for lr, r, a in itertools.product(args.lrs, args.ranks, args.alphas)
    ]

    # Если сетка из одной точки — пропускаем поиск и сразу полное обучение
    if len(grid) == 1:
        best_point = grid[0]
        log.info("Одна точка сетки — пропускаем grid search.")
    else:
        log.info("=== Фаза 1: grid search (%d точек, %d примеров) ===", len(grid), args.search_samples)
        search_train = train_ds.select(range(min(args.search_samples, len(train_ds))))
        search_eval  = eval_ds.select(range(min(500, len(eval_ds))))

        search_results: list[GridResult] = []
        for i, point in enumerate(grid, 1):
            log.info("[%d/%d] %s", i, len(grid), point.slug())
            run_dir = str(output_root / "search" / point.slug())
            result = train_one(
                point=point,
                base_model=args.base_model,
                processor=processor,
                train_ds=search_train,
                eval_ds=search_eval,
                output_dir=run_dir,
                epochs=args.search_epochs,
                batch=args.batch,
                grad_accum=args.grad_accum,
            )
            search_results.append(result)

        best_search = min(search_results, key=lambda r: r.wer)
        best_point = best_search.point
        log.info("Лучшие гиперпараметры: %s  WER=%.2f%%", best_point.slug(), best_search.wer)

        # Сохраняем сводку grid search
        summary = [
            {**asdict(r.point), "wer": r.wer, "cer": r.cer}
            for r in search_results
        ]
        (output_root / "grid_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -----------------------------------------------------------------------
    # Фаза 2: полное обучение с лучшими параметрами
    # -----------------------------------------------------------------------
    log.info("=== Фаза 2: полное обучение  %s ===", best_point.slug())
    final_result = train_one(
        point=best_point,
        base_model=args.base_model,
        processor=processor,
        train_ds=train_ds,
        eval_ds=eval_ds,
        output_dir=str(output_root / "final"),
        epochs=args.epochs,
        batch=args.batch,
        grad_accum=args.grad_accum,
    )

    # Копируем лучший адаптер в корень output
    best_dir = output_root / "best_adapter"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(final_result.adapter_dir, best_dir)

    log.info("=== Готово ===")
    log.info("WER=%.2f%%  CER=%.2f%%", final_result.wer, final_result.cer)
    log.info("Адаптер: %s", best_dir)


if __name__ == "__main__":
    main()
