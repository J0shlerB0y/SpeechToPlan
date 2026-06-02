"""Инференс FunctionGemma-270m-it: текст → JSON-задача (PlannerTask)."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.prompts import build_chat
from shared.schemas import PlannerTask

load_dotenv()
log = logging.getLogger(__name__)

# Жадный поиск JSON: первый '{' .. последний '}' (поддерживает вложенность).
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


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
    has_adapter: bool = False

    @classmethod
    def load(
        cls,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        quant: str = "int4",
        device: str = "cuda",
        adapter_path: str | None = None,
    ) -> "GemmaPlannerLLM":
        hf_token = os.getenv("HF_TOKEN")
        log.info("HF_TOKEN: %s", "set" if hf_token else "missing")

        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        adapter = adapter_path or os.getenv("LLM_ADAPTER_PATH")

        # Если адаптер был сохранён старой peft (несовпадение shape на bnb-базе) —
        # выставите LLM_FP16_FALLBACK=1, тогда база грузится в fp16 (~3 ГБ) без bnb.
        fp16_fallback = adapter and os.getenv("LLM_FP16_FALLBACK", "0") == "1"
        effective_quant = "none" if fp16_fallback else quant

        kwargs: dict = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "token": hf_token,
        }
        bnb = _bnb_config(effective_quant) if device == "cuda" else None
        if device == "cuda":
            kwargs["device_map"] = "auto"
        if bnb is not None:
            kwargs["quantization_config"] = bnb

        if effective_quant != quant:
            log.info("LLM_FP16_FALLBACK=1 → грузим базу fp16 (вместо %s) чтобы избежать shape mismatch", quant)

        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if device != "cuda":
            model.to(device)

        has_adapter = False
        if adapter:
            from peft import PeftModel
            log.info("Накатываем LoRA-адаптер: %s", adapter)
            model = PeftModel.from_pretrained(model, adapter, token=hf_token)
            has_adapter = True
        else:
            log.info("Запуск без адаптера (чистая база)")

        model.eval()
        return cls(tokenizer=tokenizer, model=model, device=device, has_adapter=has_adapter)

    # ---------- генерация ----------
    @torch.inference_mode()
    def generate_raw(self, text: str, max_new_tokens: int = 448, few_shot: bool = False) -> str:
        """Вернуть текст ответа модели. few_shot=True — для baseline без адаптера."""
        messages = build_chat(text, few_shot=few_shot)

        # apply_chat_template сам ставит <start_of_turn>model\n в конце
        # благодаря add_generation_prompt=True. НИЧЕГО после неё дописывать нельзя!
        enc = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[-1]

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][prompt_len:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        log.info(
            "generation | prompt_tokens=%d new_tokens=%d | first16=%s",
            prompt_len, gen_ids.shape[-1], gen_ids[:16].tolist(),
        )
        return completion

    def generate_json(self, text: str, max_new_tokens: int = 448) -> dict:
        completion = self.generate_raw(text, max_new_tokens=max_new_tokens)
        print(f"\n--- LLM OUTPUT ---\n{completion}\n------------------")
        return self._extract_json(completion)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Достаём JSON: markdown-fence → первый объект → весь текст."""
        text = (text or "").strip()
        if not text:
            return {"_error": "empty_output"}

        m = _JSON_FENCE.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        m = _JSON_OBJ.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_error": "invalid_json", "raw": text}

    def to_task(self, text: str) -> PlannerTask:
        data = self.generate_json(text)

        if "_error" in data:
            log.warning("LLM fallback: %s (raw=%r)", data["_error"], data.get("raw", "")[:120])
            # PlannerTask требует title — кладём короткую заглушку из входного текста.
            fallback_title = (text or "Не удалось распознать").strip()[:80]
            return PlannerTask(
                title=fallback_title,
                description=None,
                deadline=None,
                priority="medium",
            )

        try:
            return PlannerTask(**data)
        except Exception as e:  # pydantic ValidationError, KeyError, ...
            log.warning("Pydantic не принял ответ модели: %s | data=%s", e, data)
            # Спасаем то, что можем: чекпоинты пытаемся сохранить, если они валидны.
            raw_priority = data.get("priority")
            raw_checkpoints = data.get("checkpoints") or []
            safe_checkpoints = []
            if isinstance(raw_checkpoints, list):
                for c in raw_checkpoints:
                    if isinstance(c, dict) and c.get("step"):
                        safe_checkpoints.append(
                            {"step": str(c["step"]), "deadline": c.get("deadline")}
                        )
            return PlannerTask(
                title=str(data.get("title") or text or "Задача")[:80],
                description=data.get("description"),
                deadline=data.get("deadline"),
                priority=raw_priority if raw_priority in {"low", "medium", "high", "urgent"} else "medium",
                checkpoints=safe_checkpoints,
                tags=data.get("tags") or [],
            )
