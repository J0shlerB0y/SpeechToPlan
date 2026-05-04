"""Дообучение Whisper-tiny LoRA-адаптерами для русской речи.

Под RTX 3050 Ti (4 ГБ VRAM):
  * базовая модель грузится в fp16
  * включён gradient_checkpointing
  * batch_size=4, grad_accum=8 -> эффективный batch 32
  * обучаются только LoRA-адаптеры (q_proj, v_proj) — ~0.5% параметров
  * после обучения — оценка WER на validation split

Запуск:
    python -m asr_emotion.training.train_whisper_lora \
        --dataset mozilla-foundation/common_voice_17_0 \
        --lang ru \
        --output ./checkpoints/whisper-tiny-ru-lora
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Audio, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train_whisper_lora")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Whisper стартует с decoder_start_token — отрезаем его, если он попал.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _build_dataset(name: str, lang: str, split: str, max_samples: int | None):
    log.info("Загружаем %s | split=%s | lang=%s", name, split, lang)
    ds = load_dataset(name, lang, split=split, trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    keep = {"audio", "sentence"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    return ds


def _prepare_fn(processor: WhisperProcessor):
    def _prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch
    return _prepare


def _compute_metrics_fn(processor: WhisperProcessor):
    import evaluate
    wer = evaluate.load("wer")

    def _compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer.compute(predictions=pred_str, references=label_str)}
    return _compute_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="openai/whisper-tiny")
    p.add_argument("--dataset", default="mozilla-foundation/common_voice_17_0")
    p.add_argument("--lang", default="ru")
    p.add_argument("--output", default="./checkpoints/whisper-tiny-ru-lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--max-eval", type=int, default=400)
    args = p.parse_args()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model)
    tokenizer = WhisperTokenizer.from_pretrained(args.base_model, language=args.lang, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.lang, task="transcribe")

    train_ds = _build_dataset(args.dataset, args.lang, "train", args.max_train)
    eval_ds = _build_dataset(args.dataset, args.lang, "validation", args.max_eval)
    train_ds = train_ds.map(_prepare_fn(processor), remove_columns=train_ds.column_names, num_proc=2)
    eval_ds = eval_ds.map(_prepare_fn(processor), remove_columns=eval_ds.column_names, num_proc=2)

    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="auto"
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # обязательно при gradient_checkpointing

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        num_train_epochs=args.epochs,
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
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.feature_extractor,
        compute_metrics=_compute_metrics_fn(processor),
    )

    trainer.train()
    metrics = trainer.evaluate()
    log.info("Финальный WER=%.2f%%", metrics.get("eval_wer", float("nan")))

    # Сохраняем LoRA-адаптер. Базовую модель не трогаем — её можно скачать снова.
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    log.info("Адаптер сохранён в %s", args.output)


if __name__ == "__main__":
    main()
