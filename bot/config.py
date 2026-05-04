"""Конфигурация бота. Все переменные окружения читаются здесь."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BotConfig:
    token: str
    tmp_dir: Path
    asr_model_size: str = "tiny"          # tiny / base — для 4 ГБ VRAM
    asr_compute_type: str = "int8"        # int8 / int8_float16
    emotion_model: str = "superb/wav2vec2-base-superb-er"
    llm_model_path: str = "google/functiongemma-270m-it"
    llm_quant: str = "int4"               # int4 / int8
    device: str = "cuda"                  # cuda / cpu
    max_audio_seconds: int = 120


def load_config() -> BotConfig:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в окружении")
    tmp_dir = Path(os.getenv("BOT_TMP_DIR", "./tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return BotConfig(
        token=token,
        tmp_dir=tmp_dir,
        asr_model_size=os.getenv("ASR_MODEL_SIZE", "tiny"),
        asr_compute_type=os.getenv("ASR_COMPUTE_TYPE", "int8"),
        emotion_model=os.getenv("EMOTION_MODEL", "superb/wav2vec2-base-superb-er"),
        llm_model_path=os.getenv("LLM_MODEL_PATH", "google/functiongemma-270m-it"),
        llm_quant=os.getenv("LLM_QUANT", "int4"),
        device=os.getenv("DEVICE", "cuda"),
        max_audio_seconds=int(os.getenv("MAX_AUDIO_SECONDS", "120")),
    )
