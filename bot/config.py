from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BotConfig:
    token: str
    tmp_dir: Path
    ml_service_url: str
    ml_request_timeout: float = 120.0
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
        ml_service_url=os.getenv("ML_SERVICE_URL", "http://host.docker.internal:8000"),
        ml_request_timeout=float(os.getenv("ML_REQUEST_TIMEOUT", "120")),
        max_audio_seconds=int(os.getenv("MAX_AUDIO_SECONDS", "120")),
    )
