"""Speech Emotion Recognition (SER) — лёгкий wav2vec2 классификатор.

Используется модель `superb/wav2vec2-base-superb-er` (~95M параметров) — fp16
в режиме eval умещается в ~250 МБ VRAM, что приемлемо для RTX 3050 Ti параллельно
с whisper-tiny INT8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from shared.schemas import Emotion, EmotionResult


# Карта меток superb/wav2vec2-base-superb-er → наши эмоции
SUPERB_TO_EMOTION: Mapping[str, Emotion] = {
    "neu": Emotion.NEUTRAL,
    "hap": Emotion.HAPPY,
    "sad": Emotion.SAD,
    "ang": Emotion.ANGRY,
    "neutral": Emotion.NEUTRAL,
    "happy": Emotion.HAPPY,
    "sadness": Emotion.SAD,
    "anger": Emotion.ANGRY,
    "fear": Emotion.ANXIOUS,
    "surprise": Emotion.URGENT,
}

URGENT_LABELS = {Emotion.ANGRY, Emotion.ANXIOUS, Emotion.URGENT}


@dataclass
class EmotionRecognizer:
    extractor: AutoFeatureExtractor
    model: AutoModelForAudioClassification
    device: str

    @classmethod
    def load(cls, model_id: str, device: str = "cuda") -> "EmotionRecognizer":
        extractor = AutoFeatureExtractor.from_pretrained(model_id)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForAudioClassification.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device).eval()
        return cls(extractor=extractor, model=model, device=device)

    @torch.inference_mode()
    def predict(self, audio_16k: np.ndarray) -> EmotionResult:
        inputs = self.extractor(
            audio_16k, sampling_rate=16_000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits.float(), dim=-1)
        idx = int(torch.argmax(probs))
        score = float(probs[idx])
        raw_label = self.model.config.id2label[idx].lower()
        emotion = SUPERB_TO_EMOTION.get(raw_label, Emotion.NEUTRAL)
        return EmotionResult(
            label=emotion,
            score=score,
            is_urgent=emotion in URGENT_LABELS and score > 0.45,
        )
