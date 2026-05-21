"""Инференс FunctionGemma-270m-it через transformers + bitsandbytes (4-bit/8-bit).

Под RTX 3050 Ti 4 ГБ:
  * INT4 NF4 + double quant — модель занимает ~250 МБ VRAM
  * параллельно влезает whisper-tiny INT8 + wav2vec2 SER (≈1.5 ГБ суммарно)
  * генерация greedy + json.loads валидация ответа

Если LoRA-адаптер дообучен (см. training/qlora_grid_search.py) — указывай его
через переменную окружения LLM_ADAPTER_PATH.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.prompts import build_chat
from shared.schemas import EnrichedUtterance, PlannerTask

log = logging.getLogger(__name__)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _bnb_config(quant: str):
    """Импорт bitsandbytes ленивый: на Windows/CPU он часто не ставится,
    а для quant='none' он и не нужен."""
    if quant not in {"int4", "int8"}:
        return None
    from transformers import BitsAndBytesConfig  # noqa: WPS433

    if quant == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


@dataclass
class GemmaPlannerLLM:
    tokenizer: Any
    model: Any
    device: str

    @classmethod
    def load(
        cls,
        model_path: str = "google/functiongemma-270m-it",
        quant: str = "int4",
        device: str = "cuda",
        adapter_path: str | None = None,
    ) -> "GemmaPlannerLLM":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        bnb = _bnb_config(quant) if device == "cuda" else None

        kwargs: dict = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        if device == "cuda":
            kwargs["device_map"] = "auto"
        if bnb is not None:
            kwargs["quantization_config"] = bnb

        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if device != "cuda":
            model.to(device)

        adapter = adapter_path or os.getenv("LLM_ADAPTER_PATH")
        if adapter:
            from peft import PeftModel
            log.info("Накатываем LoRA адаптер: %s", adapter)
            model = PeftModel.from_pretrained(model, adapter)

        model.eval()
        return cls(tokenizer=tokenizer, model=model, device=device)

    @torch.inference_mode()
    def generate_json(self, user_prompt: str, max_new_tokens: int = 256) -> dict:
        messages = build_chat(user_prompt)
        enc = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0][enc["input_ids"].shape[-1]:]
        completion = self.tokenizer.decode(gen, skip_special_tokens=True)
        return self._extract_json(completion)

    @staticmethod
    def _extract_json(text: str) -> dict:
        m = JSON_RE.search(text)
        if not m:
            raise ValueError(f"Модель не вернула JSON: {text!r}")
        return json.loads(m.group(0))

    def to_task(self, utterance: EnrichedUtterance) -> PlannerTask:
        prompt = utterance.to_prompt()
        try:
            data = self.generate_json(prompt)
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("LLM вернула невалидный JSON: %s — fallback", e)
            data = {
                "title": utterance.text[:80] or "Новая задача",
                "priority": "high" if utterance.emotion.is_urgent else "medium",
                "tags": [],
                "raw_emotion": utterance.emotion.label.value,
            }
        # Pydantic валидирует контракт.
        return PlannerTask(**data)
