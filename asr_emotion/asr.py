"""Обёртка над faster-whisper. INT8 на CPU/GPU — минимум VRAM."""
from __future__ import annotations

from dataclasses import dataclass

from faster_whisper import WhisperModel

from shared.schemas import ASRResult


@dataclass
class WhisperASR:
    model: WhisperModel

    @classmethod
    def load(
        cls,
        size: str = "tiny",
        device: str = "cuda",
        compute_type: str = "int8",
        local_path: str | None = None,
    ) -> "WhisperASR":
        # local_path позволяет указать каталог с дообученной CT2-моделью.
        model = WhisperModel(
            local_path or size,
            device=device,
            compute_type=compute_type,
        )
        return cls(model=model)

    def transcribe(self, audio_path: str, language: str = "ru") -> ASRResult:
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        chunks: list[str] = []
        logprobs: list[float] = []
        for s in segments:
            chunks.append(s.text)
            logprobs.append(s.avg_logprob)
        return ASRResult(
            text=" ".join(c.strip() for c in chunks).strip(),
            language=info.language,
            duration_sec=info.duration,
            avg_logprob=sum(logprobs) / max(1, len(logprobs)),
        )
