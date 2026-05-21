"""Чтение аудио → float32 моно 16 кГц.

Используем PyAV (тащится транзитивно с faster-whisper), чтобы не зависеть
от наличия ffmpeg в системном PATH (актуально для Windows-разработки).
"""
from __future__ import annotations

import numpy as np

TARGET_SR = 16_000


def load_audio_16k(path: str) -> np.ndarray:
    """Декодирует ogg/mp3/wav и любой формат, поддерживаемый ffmpeg-libs внутри PyAV.

    Возвращает np.float32, моно, 16 кГц, в диапазоне [-1, 1].
    """
    import av

    container = av.open(path)
    try:
        stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration as e:
        raise ValueError(f"В файле {path!r} нет аудио-потока") from e

    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=TARGET_SR,
    )

    pcm_chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()  # shape (1, N) для mono s16
            pcm_chunks.append(arr.reshape(-1))

    # Хвостовой flush ресемплера.
    for resampled in resampler.resample(None):
        arr = resampled.to_ndarray()
        pcm_chunks.append(arr.reshape(-1))

    container.close()

    if not pcm_chunks:
        return np.zeros(0, dtype=np.float32)

    pcm = np.concatenate(pcm_chunks).astype(np.float32) / 32768.0
    return pcm
