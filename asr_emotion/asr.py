"""ASR-обёртки.

Два backend-а:

* `WhisperASR` (faster-whisper / CTranslate2) — быстрый INT8/INT8_float16,
  но НЕ умеет применять PEFT-LoRA-адаптер. Используется по умолчанию.

* `WhisperASRTransformers` (HF transformers + PEFT) — медленнее, но именно сюда
  накатывается дообученный LoRA-адаптер из
  `checkpoints/whisper-lora-ru/best_adapter/`.

В ml_service переключение через `ML_ASR_BACKEND={faster|transformers}` +
`ML_ASR_ADAPTER=<путь до adapter dir>`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from shared.schemas import ASRResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CTranslate2-backend (быстрый, без LoRA)
# ---------------------------------------------------------------------------
@dataclass
class WhisperASR:
    model: Any

    @classmethod
    def load(
        cls,
        size: str = "tiny",
        device: str = "cuda",
        compute_type: str = "int8",
        local_path: str | None = None,
    ) -> "WhisperASR":
        from faster_whisper import WhisperModel
        model = WhisperModel(local_path or size, device=device, compute_type=compute_type)
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


# ---------------------------------------------------------------------------
# transformers + PEFT backend (медленнее, но применяет LoRA-адаптер)
# ---------------------------------------------------------------------------
@dataclass
class WhisperASRTransformers:
    model: Any
    processor: Any
    device: str
    language: str = "ru"
    has_adapter: bool = False
    _resampler_cache: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        size: str = "openai/whisper-tiny",
        device: str = "cuda",
        compute_type: str = "float16",
        adapter_path: str | None = None,
        language: str = "ru",
    ) -> "WhisperASRTransformers":
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        # Удобный alias: "tiny" → "openai/whisper-tiny"
        if "/" not in size:
            size = f"openai/whisper-{size}"

        log.info("Грузим базу Whisper: %s", size)
        processor = WhisperProcessor.from_pretrained(size, language=language, task="transcribe")
        dtype = torch.float16 if (device == "cuda" and compute_type != "float32") else torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(size, torch_dtype=dtype)
        model.to(device).eval()

        # forced_decoder_ids нужны чтобы Whisper писал по-русски без авто-детекта.
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )
        model.config.suppress_tokens = []

        has_adapter = False
        if adapter_path:
            from peft import PeftModel
            log.info("Накатываем Whisper LoRA-адаптер: %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload() if False else model
            # merge_and_unload даёт чуть быстрее inference, но ломает torch_dtype под 4 ГБ.
            model.eval()
            has_adapter = True
        else:
            log.info("Whisper-адаптер не задан — чистая база")

        return cls(
            model=model, processor=processor, device=device,
            language=language, has_adapter=has_adapter,
        )

    def _read_audio_16k(self, path: str):
        """Декодирует любой контейнер в float32 моно 16 кГц через PyAV."""
        import av
        import numpy as np

        container = av.open(path)
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16_000)

        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for r in resampler.resample(frame):
                chunks.append(r.to_ndarray().reshape(-1))
        for r in resampler.resample(None):
            chunks.append(r.to_ndarray().reshape(-1))
        container.close()
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0

    def transcribe(self, audio_path: str, language: str | None = None) -> ASRResult:
        import torch

        lang = language or self.language
        audio = self._read_audio_16k(audio_path)
        duration = len(audio) / 16_000

        inputs = self.processor(
            audio, sampling_rate=16_000, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.model.device, dtype=self.model.dtype)

        with torch.inference_mode():
            generated = self.model.generate(
                input_features,
                language=lang,
                task="transcribe",
                num_beams=1,
                max_new_tokens=225,
            )

        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return ASRResult(text=text, language=lang, duration_sec=duration, avg_logprob=0.0)
