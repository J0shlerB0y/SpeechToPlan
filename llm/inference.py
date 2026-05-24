"""Инференс FunctionGemma-270m-it — без эмоций, только текст → JSON-задача."""
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
from shared.schemas import PlannerTask

log = logging.getLogger(__name__)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _bnb_config(quant: str):
    if quant not in {"int4", "int8"}:
        return None
    from transformers import BitsAndBytesConfig

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

        from dotenv import load_dotenv
        load_dotenv()
        print("TOKEN:", os.getenv("HF_TOKEN"))
        kwargs: dict = {
            "token": os.getenv("HF_TOKEN"),
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        bnb = _bnb_config(quant) if device == "cuda" else None
        if device == "cuda":
            kwargs["device_map"] = "auto"
        if bnb is not None:
            kwargs["quantization_config"] = bnb

        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if device != "cuda":
            model.to(device)

        adapter = adapter_path or os.getenv("LLM_ADAPTER_PATH")
        if adapter:
            from peft import PeftModel
            log.info("Накатываем LoRA адаптер: %s", adapter)
            model = PeftModel.from_pretrained(model, adapter, **kwargs)

        model.eval()
        return cls(tokenizer=tokenizer, model=model, device=device)

    @torch.inference_mode()
    def generate_json(self, text: str, max_new_tokens: int = 256) -> dict:
        messages = build_chat(text)
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

    def to_task(self, text: str) -> PlannerTask:
        """Принимает чистый текст (после Whisper), возвращает PlannerTask."""
        try:
            data = self.generate_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("LLM вернула невалидный JSON: %s — fallback", e)
            data = {
                "title": text[:80] or "Новая задача",
                "description": None,
                "deadline": None,
                "priority": "medium",
            }
        return PlannerTask(**data)