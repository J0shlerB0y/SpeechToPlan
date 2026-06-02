"""FastAPI-сервис: оборачивает ML-пайплайн в HTTP API.

Запуск (на хосте, с GPU):
    set ML_DEVICE=cuda
    set ML_LLM_QUANT=int4
    uvicorn ml_service.server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                 — readiness probe + что подгружено
    POST /process/text           — body: {"text": "..."}            → PlannerTask JSON
    POST /process/voice          — multipart: file=<.ogg/.wav/.mp3> → PlannerTask JSON

ENV (читаются здесь):
    ML_DEVICE         cuda|cpu                       (default: cuda)
    ML_ASR_BACKEND    faster|transformers            (default: faster)
    ML_ASR_SIZE       tiny|base|small|...            (default: tiny)
    ML_ASR_COMPUTE    int8|float16|float32           (default: int8)
    ML_ASR_ADAPTER    <path>                         (требует backend=transformers)
    ML_LLM_MODEL      HF id                          (default: google/functiongemma-270m-it)
    ML_LLM_QUANT      int4|int8|none                 (default: int4)
    ML_LLM_ADAPTER    <path>                         (LoRA адаптер для LLM)
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from asr_emotion.asr import WhisperASR, WhisperASRTransformers
from llm.inference import GemmaPlannerLLM
from shared.schemas import PlannerTask

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("ml_service")


class _State:
    asr: "WhisperASR | WhisperASRTransformers | None" = None
    llm: GemmaPlannerLLM | None = None
    lock: asyncio.Lock | None = None
    asr_backend: str = "faster"
    asr_has_adapter: bool = False


state = _State()


def _load_models() -> None:
    device       = os.getenv("ML_DEVICE", "cuda")
    asr_backend  = os.getenv("ML_ASR_BACKEND", "faster").lower()
    asr_size     = os.getenv("ML_ASR_SIZE", "tiny")
    asr_compute  = os.getenv("ML_ASR_COMPUTE", "int8")
    asr_adapter  = os.getenv("ML_ASR_ADAPTER") or None
    llm_model    = os.getenv("ML_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    llm_quant    = os.getenv("ML_LLM_QUANT", "int4")
    llm_adapter  = os.getenv("ML_LLM_ADAPTER") or None

    state.asr_backend = asr_backend
    log.info("ASR backend = %s | size=%s | compute=%s | adapter=%s",
             asr_backend, asr_size, asr_compute, asr_adapter)

    if asr_backend == "transformers" or asr_adapter:
        # Только этот backend умеет PEFT-адаптеры.
        if asr_backend == "faster" and asr_adapter:
            log.warning(
                "ML_ASR_ADAPTER задан, но backend=faster — faster-whisper не умеет LoRA."
                " Переключаемся на transformers backend."
            )
        state.asr = WhisperASRTransformers.load(
            size=asr_size,
            device=device,
            compute_type=asr_compute,
            adapter_path=asr_adapter,
        )
        state.asr_has_adapter = state.asr.has_adapter
    else:
        state.asr = WhisperASR.load(
            size=asr_size, device=device, compute_type=asr_compute,
        )

    log.info("Загружаем LLM %s (quant=%s, adapter=%s)",
             llm_model, llm_quant, llm_adapter)
    state.llm = GemmaPlannerLLM.load(
        model_path=llm_model, quant=llm_quant, device=device, adapter_path=llm_adapter,
    )
    log.info("ML-сервис готов | asr=%s/has_adapter=%s | llm/has_adapter=%s",
             type(state.asr).__name__, state.asr_has_adapter, state.llm.has_adapter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.lock = asyncio.Lock()
    await asyncio.get_running_loop().run_in_executor(None, _load_models)
    yield


app = FastAPI(title="SpeechToPlan ML service", version="0.2.0", lifespan=lifespan)


class TextRequest(BaseModel):
    text: str


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if state.asr and state.llm else "loading",
        "asr_loaded": state.asr is not None,
        "asr_backend": state.asr_backend,
        "asr_has_lora": state.asr_has_adapter,
        "llm_loaded": state.llm is not None,
        "llm_has_lora": bool(state.llm and state.llm.has_adapter),
    }


@app.post("/process/text", response_model=PlannerTask)
async def process_text(req: TextRequest) -> PlannerTask:
    if not state.llm:
        raise HTTPException(503, "LLM ещё не готова")
    async with state.lock:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, state.llm.to_task, req.text)


@app.post("/process/voice", response_model=PlannerTask)
async def process_voice(file: UploadFile = File(...)) -> PlannerTask:
    if not (state.asr and state.llm):
        raise HTTPException(503, "Модели ещё не готовы")

    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        async with state.lock:
            loop = asyncio.get_running_loop()
            asr_result = await loop.run_in_executor(None, state.asr.transcribe, str(tmp_path))
            log.info("ASR (%.1fс): %r", asr_result.duration_sec, asr_result.text)
            return await loop.run_in_executor(None, state.llm.to_task, asr_result.text)
    finally:
        tmp_path.unlink(missing_ok=True)
