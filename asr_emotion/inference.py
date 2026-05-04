"""Вариант Б: ASR и SER крутятся ПАРАЛЛЕЛЬНО, после чего результат склеивается.

Параллелизм через thread-pool: faster-whisper работает в C++/CTranslate2,
освобождает GIL — это даёт реальный overlap по времени с torch-моделью SER.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from asr_emotion.asr import WhisperASR
from asr_emotion.audio_utils import load_audio_16k
from asr_emotion.emotion import EmotionRecognizer
from shared.schemas import EnrichedUtterance

logger = logging.getLogger(__name__)


@dataclass
class AsrEmotionPipeline:
    asr: WhisperASR
    ser: EmotionRecognizer
    pool: ThreadPoolExecutor

    @classmethod
    def load(
        cls,
        whisper_size: str = "tiny",
        compute_type: str = "int8",
        emotion_model_id: str = "superb/wav2vec2-base-superb-er",
        device: str = "cuda",
    ) -> "AsrEmotionPipeline":
        asr = WhisperASR.load(size=whisper_size, device=device, compute_type=compute_type)
        ser = EmotionRecognizer.load(emotion_model_id, device=device)
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr-ser")
        return cls(asr=asr, ser=ser, pool=pool)

    def transcribe_with_emotion(self, audio_path: str) -> EnrichedUtterance:
        # Декодируем один раз — отдадим оба массива параллельным задачам.
        audio_np = load_audio_16k(audio_path)

        fut_asr = self.pool.submit(self.asr.transcribe, audio_path)
        fut_ser = self.pool.submit(self.ser.predict, audio_np)

        asr_res = fut_asr.result()
        ser_res = fut_ser.result()
        logger.debug("ASR=%s | SER=%s", asr_res, ser_res)

        return EnrichedUtterance(text=asr_res.text, emotion=ser_res, source="voice")

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
