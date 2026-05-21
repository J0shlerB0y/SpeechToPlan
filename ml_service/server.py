"""FastAPI-сервис: оборачивает ML-пайплайн в HTTP API.

Запуск (на хосте, с GPU):
    set ML_DEVICE=cuda
    set ML_LLM_QUANT=int4
    uvicorn ml_service.server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                 — readiness probe
    POST /process/text           — body: {"text": "..."}            → PlannerTask JSON
    POST /process/voice          — multipart: file=<.ogg/.wav/.mp3> → PlannerTask JSON

Бот в Docker обращается сюда через http://host.docker.internal:8000.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from asr_emotion.inference import AsrEmotionPipeline
from llm.inference import GemmaPlannerLLM
from shared.schemas import Emotion, EmotionResult, EnrichedUtterance, PlannerTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("ml_service")


class _State:
    asr: AsrEmotionPipeline | None = None
    llm: GemmaPlannerLLM | None = None
    lock: asyncio.Lock | None = None


state = _State()


def _load_models() -> None:
    device = os.getenv("ML_DEVICE", "cuda")
    asr_size = os.getenv("ML_ASR_SIZE", "tiny")
    asr_compute = os.getenv("ML_ASR_COMPUTE", "int8")
    emotion_model = os.getenv("ML_EMOTION_MODEL", "superb/wav2vec2-base-superb-er")
    llm_model = os.getenv("ML_LLM_MODEL", "google/functiongemma-270m-it")
    llm_quant = os.getenv("ML_LLM_QUANT", "int4")
    adapter = os.getenv("ML_LLM_ADAPTER") or None

    log.info("Загружаем ASR+SER (whisper=%s/%s, emotion=%s) на %s",
             asr_size, asr_compute, emotion_model, device)
    state.asr = AsrEmotionPipeline.load(
        whisper_size=asr_size,
        compute_type=asr_compute,
        emotion_model_id=emotion_model,
        device=device,
    )
    log.info("Загружаем LLM %s (quant=%s, adapter=%s) на %s",
             llm_model, llm_quant, adapter, device)
    state.llm = GemmaPlannerLLM.load(
        model_path=llm_model, quant=llm_quant, device=device, adapter_path=adapter,
    )
    log.info("ML-сервис готов.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.lock = asyncio.Lock()
    # Загрузка моделей блокирующая — выносим в thread, чтобы не держать loop.
    await asyncio.get_running_loop().run_in_executor(None, _load_models)
    try:
        yield
    finally:
        if state.asr is not None:
            state.asr.shutdown()


app = FastAPI(title="SpeechToPlan ML service", version="0.1.0", lifespan=lifespan)


class TextRequest(BaseModel):
    text: str


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if state.asr and state.llm else "loading",
        "asr": state.asr is not None,
        "llm": state.llm is not None,
    }


@app.post("/process/text", response_model=PlannerTask)
async def process_text(req: TextRequest) -> PlannerTask:
    if not state.llm:
        raise HTTPException(503, "LLM ещё не готова")
    enriched = EnrichedUtterance(
        text=req.text,
        emotion=EmotionResult(label=Emotion.NEUTRAL, score=1.0, is_urgent=False),
        source="text",
    )
    async with state.lock:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, state.llm.to_task, enriched)


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
            enriched = await loop.run_in_executor(
                None, state.asr.transcribe_with_emotion, str(tmp_path)
            )
            log.info("ASR+SER: %s", enriched.to_prompt())
            return await loop.run_in_executor(None, state.llm.to_task, enriched)
    finally:
        tmp_path.unlink(missing_ok=True)
