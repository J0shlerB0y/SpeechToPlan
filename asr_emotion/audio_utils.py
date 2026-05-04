"""Чтение и ресемплинг аудио до 16 кГц моно — общее для ASR и SER моделей."""
from __future__ import annotations

import numpy as np


def load_audio_16k(path: str) -> np.ndarray:
    """Декодирует ogg/mp3/wav → float32 моно 16 кГц через ffmpeg."""
    import subprocess

    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-ar", "16000",
        "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    return audio
