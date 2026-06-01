"""Smoke-test пайплайна без Telegram-бота.

Примеры:
    # чистая база (без LoRA адаптеров)
    python -m scripts.test_pipeline ".\\тестовое.ogg" --device cpu --quant none

    # с LoRA-адаптером для LLM
    python -m scripts.test_pipeline ".\\тестовое.ogg" --device cpu --quant none \
        --llm-adapter .\\checkpoints\\gemma-grid\\best_adapter

    # с LoRA-адаптером для Whisper (требует --asr-backend transformers)
    python -m scripts.test_pipeline ".\\тестовое.ogg" --device cuda \
        --asr-backend transformers \
        --asr-adapter .\\checkpoints\\whisper-lora-ru\\search\\lr1e-03_r8_a16\\adapter

    # текст напрямую, без ASR
    python -m scripts.test_pipeline --text "завтра в 9 утра встреча"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("smoke")


def main() -> int:
    p = argparse.ArgumentParser(description="SpeechToPlan smoke test")
    p.add_argument("audio", nargs="?", help="путь до .ogg / .wav / .mp3")
    p.add_argument("--text", help="подать текст напрямую, без ASR")
    p.add_argument("--device",       default=os.getenv("DEVICE", "cuda"), choices=["cuda", "cpu"])
    p.add_argument("--quant",        default=os.getenv("LLM_QUANT", "int4"), choices=["int4", "int8", "none"])
    p.add_argument("--asr-backend",  default=os.getenv("ASR_BACKEND", "faster"), choices=["faster", "transformers"])
    p.add_argument("--asr-size",     default=os.getenv("ASR_MODEL_SIZE", "tiny"))
    p.add_argument("--asr-compute",  default=os.getenv("ASR_COMPUTE_TYPE", "int8"))
    p.add_argument("--asr-adapter",  default=os.getenv("ASR_ADAPTER_PATH"))
    p.add_argument("--llm-model",    default=os.getenv("LLM_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"))
    p.add_argument("--llm-adapter",  default=os.getenv("LLM_ADAPTER_PATH"))
    p.add_argument("--skip-llm",     action="store_true", help="только ASR, без LLM")
    args = p.parse_args()

    if not args.audio and not args.text:
        p.error("Укажи аудио-файл или --text")

    text: str | None = None

    # --- ASR ---
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            log.error("Файл не найден: %s", audio_path)
            return 2

        if args.asr_backend == "transformers" or args.asr_adapter:
            from asr_emotion.asr import WhisperASRTransformers
            log.info("ASR backend=transformers, size=%s, adapter=%s", args.asr_size, args.asr_adapter)
            t0 = time.perf_counter()
            asr = WhisperASRTransformers.load(
                size=args.asr_size, device=args.device,
                compute_type=args.asr_compute,
                adapter_path=args.asr_adapter,
            )
            log.info("ASR загружен за %.1fс | has_adapter=%s", time.perf_counter() - t0, asr.has_adapter)
        else:
            from asr_emotion.asr import WhisperASR
            log.info("ASR backend=faster-whisper, size=%s/%s", args.asr_size, args.asr_compute)
            t0 = time.perf_counter()
            asr = WhisperASR.load(size=args.asr_size, device=args.device, compute_type=args.asr_compute)
            log.info("ASR загружен за %.1fс", time.perf_counter() - t0)

        t1 = time.perf_counter()
        asr_res = asr.transcribe(str(audio_path))
        log.info("ASR inference: %.1fс | язык=%s", time.perf_counter() - t1, asr_res.language)
        text = asr_res.text
    else:
        text = args.text

    print(f"\n=== ASR РЕЗУЛЬТАТ ===\n{text}\n")

    if args.skip_llm:
        return 0

    # --- LLM ---
    from llm.inference import GemmaPlannerLLM
    log.info("Загружаем LLM %s (quant=%s, adapter=%s)", args.llm_model, args.quant, args.llm_adapter)
    t2 = time.perf_counter()
    llm = GemmaPlannerLLM.load(
        model_path=args.llm_model,
        quant=args.quant,
        device=args.device,
        adapter_path=args.llm_adapter,
    )
    log.info("LLM загружен за %.1fс | has_adapter=%s", time.perf_counter() - t2, llm.has_adapter)

    t3 = time.perf_counter()
    task = llm.to_task(text)
    log.info("LLM inference: %.1fс", time.perf_counter() - t3)

    print("=== PLANNER TASK ===")
    print(json.dumps(task.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
