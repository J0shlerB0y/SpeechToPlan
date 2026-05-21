"""Smoke-test всего пайплайна без Telegram-бота.

Прогоняет один аудио-файл через AsrEmotionPipeline + GemmaPlannerLLM
и печатает итоговую задачу в JSON. Полезно:
  * для проверки моделей до запуска бота;
  * для CI/smoke-теста в Docker-контейнере (см. docker-compose.test.yml).

Запуск:
    python -m scripts.test_pipeline path/to/audio.ogg --device cpu --quant none

ENV (если флаги не заданы — берутся отсюда):
    DEVICE        cuda | cpu        (default: cuda)
    LLM_QUANT     int4 | int8 | none (default: int4)
    ASR_MODEL_SIZE  tiny | base ...  (default: tiny)
    ASR_COMPUTE_TYPE  int8 | float16 ... (default: int8)
    EMOTION_MODEL  HF model id
    LLM_MODEL_PATH  HF model id
    LLM_ADAPTER_PATH  optional LoRA adapter
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Делаем корень проекта importable, если скрипт запущен напрямую.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("smoke")


def _import_pipelines():
    """Импорт тяжёлых модулей внутри функции, чтобы --help работал без torch."""
    from asr_emotion.inference import AsrEmotionPipeline
    from llm.inference import GemmaPlannerLLM
    return AsrEmotionPipeline, GemmaPlannerLLM


def main() -> int:
    p = argparse.ArgumentParser(description="SpeechToPlan smoke test")
    p.add_argument("audio", help="путь до .ogg / .wav / .mp3")
    p.add_argument("--device", default=os.getenv("DEVICE", "cuda"), choices=["cuda", "cpu"])
    p.add_argument("--quant", default=os.getenv("LLM_QUANT", "int4"),
                   choices=["int4", "int8", "none"])
    p.add_argument("--asr-size", default=os.getenv("ASR_MODEL_SIZE", "tiny"))
    p.add_argument("--asr-compute", default=os.getenv("ASR_COMPUTE_TYPE", "int8"))
    p.add_argument("--emotion-model", default=os.getenv("EMOTION_MODEL", "superb/wav2vec2-base-superb-er"))
    p.add_argument("--llm-model", default=os.getenv("LLM_MODEL_PATH", "google/functiongemma-270m-it"))
    p.add_argument("--skip-llm", action="store_true",
                   help="не загружать LLM (только ASR+эмоция)")
    p.add_argument("--skip-emotion", action="store_true",
                   help="не загружать SER (только ASR -> LLM)")
    args = p.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        log.error("Файл не найден: %s", audio_path)
        return 2

    AsrEmotionPipeline, GemmaPlannerLLM = _import_pipelines()

    # --- ASR + SER ---
    log.info("Загружаем ASR (%s / %s) и SER (%s) на %s",
             args.asr_size, args.asr_compute, args.emotion_model, args.device)
    t0 = time.perf_counter()
    if args.skip_emotion:
        from asr_emotion.asr import WhisperASR
        from shared.schemas import Emotion, EmotionResult, EnrichedUtterance
        asr = WhisperASR.load(size=args.asr_size, device=args.device,
                              compute_type=args.asr_compute)
        log.info("ASR загружен за %.1fс", time.perf_counter() - t0)
        t1 = time.perf_counter()
        asr_res = asr.transcribe(str(audio_path))
        log.info("ASR: %.1fс | язык=%s | text=%r",
                 time.perf_counter() - t1, asr_res.language, asr_res.text)
        enriched = EnrichedUtterance(
            text=asr_res.text,
            emotion=EmotionResult(label=Emotion.NEUTRAL, score=1.0, is_urgent=False),
            source="voice",
        )
    else:
        pipe = AsrEmotionPipeline.load(
            whisper_size=args.asr_size,
            compute_type=args.asr_compute,
            emotion_model_id=args.emotion_model,
            device=args.device,
        )
        log.info("ASR+SER загружены за %.1fс", time.perf_counter() - t0)
        t1 = time.perf_counter()
        enriched = pipe.transcribe_with_emotion(str(audio_path))
        log.info("ASR+SER inference: %.1fс", time.perf_counter() - t1)

    print("\n=== ASR + EMOTION ===")
    print(json.dumps({
        "text": enriched.text,
        "emotion": enriched.emotion.label.value,
        "emotion_score": round(enriched.emotion.score, 3),
        "is_urgent": enriched.emotion.is_urgent,
        "prompt_to_llm": enriched.to_prompt(),
    }, ensure_ascii=False, indent=2))

    if args.skip_llm:
        return 0

    # --- LLM ---
    log.info("Загружаем LLM %s (quant=%s)", args.llm_model, args.quant)
    t2 = time.perf_counter()
    llm = GemmaPlannerLLM.load(model_path=args.llm_model, quant=args.quant, device=args.device)
    log.info("LLM загружен за %.1fс", time.perf_counter() - t2)

    t3 = time.perf_counter()
    task = llm.to_task(enriched)
    log.info("LLM inference: %.1fс", time.perf_counter() - t3)

    print("\n=== PLANNER TASK ===")
    print(json.dumps(task.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
