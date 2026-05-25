"""Инференс FunctionGemma-270m-it — без эмоций, только текст → JSON-задача."""
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
        else:
            log.info("запуск без адаптераЮ путь, %s", adapter)
        model.eval()
        return cls(tokenizer=tokenizer, model=model, device=device)

    @torch.inference_mode()
    def generate_json(self, text: str, max_new_tokens: int = 256) -> dict:
        # ПЕРЕД ГЕНЕРАЦИЕЙ: Убедимся, что модель в режиме оценки
        self.model.eval()

        messages = build_chat(text)

        # Используем tokenize=False для контроля над текстом
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Добавляем { только если мы уверены, что модель молчит без этого
        # prompt += "{"

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # СТАВИМ FALSE ОБРАТНО
                # repetition_penalty лучше пока убрать или поставить 1.0
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            gen = out[0][enc["input_ids"].shape[-1]:]
            completion = self.tokenizer.decode(gen, skip_special_tokens=True)

            # Если мы добавляли { в промпт вручную, приклеиваем её здесь
            # completion = "{" + completion

            print(f"\n--- LLM OUTPUT ---\n{completion}\n------------------")
            return self._extract_json(completion)

        except Exception as e:
            print(f"Критическая ошибка генерации: {e}")
            return {"error": "generation_failed"}

    def _extract_json(self, text: str) -> dict:
        """Метод для очистки ответа модели от мусора и парсинга JSON."""
        import json
        import re

        try:
            # 1. Ищем JSON блоки (модель часто пишет ```json ... ```)
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 2. Если блоков нет, ищем просто что-то похожее на структуру { }
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 3. Крайний случай — пробуем парсить строку целиком
            return json.loads(text.strip())
        except (json.JSONDecodeError, AttributeError):
            # Если всё упало, возвращаем хотя бы сырой текст, чтобы не валить сервер
            return {"error": "invalid_json", "raw_text": text}

    def to_task(self, text: str) -> "PlannerTask":  # Или как там у тебя импортируется PlannerTask
        data = self.generate_json(text)

        # Если наша функция-парсер вернула ошибку, возвращаем пустую/дефолтную задачу
        if "error" in data:
            print(f"ОШИБКА LLM: Не удалось распарсить JSON. Сырой текст: '{data.get('raw_text')}'")
            # Возвращаем заглушку, чтобы сервер не падал с 500 ошибкой
            return PlannerTask(title="Не удалось распознать план",
                               is_error=True)  # Добавь поле is_error в Pydantic, если его нет

        try:
            return PlannerTask(**data)
        except Exception as e:
            # На случай если JSON валидный, но не хватает полей для Pydantic
            print(f"ОШИБКА Pydantic: {e}. Данные: {data}")
            return PlannerTask(title="Ошибка валидации плана")