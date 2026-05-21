# SpeechToPlan

Локальный Telegram-ассистент-планировщик: голос/текст → структурированная JSON-задача.

```
 ┌──────────────────────────┐  HTTP   ┌──────────────────────────────────┐
 │ Telegram-бот (Docker)    │ ─────►  │ ML-сервис (хост, GPU)            │
 │ aiogram 3.x + httpx      │ ◄─────  │ FastAPI + faster-whisper +       │
 │                          │ JSON    │ wav2vec2 SER + FunctionGemma     │
 └──────────────────────────┘         └──────────────────────────────────┘
                       host.docker.internal:8000
```

ML-стек запускается на хосте (прямой доступ к GPU без `nvidia-container-toolkit`),
бот — в лёгком Docker-образе. Между ними FastAPI.

## Структура
- `bot/` — aiogram-бот, обращается к ML по HTTP
- `ml_service/` — FastAPI-сервис: оборачивает asr_emotion + llm
- `asr_emotion/` — параллельный ASR (faster-whisper) + SER (wav2vec2)
- `llm/` — FunctionGemma-270m-it: инференс и QLoRA Grid Search
- `shared/` — Pydantic-контракты (`EnrichedUtterance`, `PlannerTask`)
- `scripts/test_pipeline.py` — smoke-тест pipeline без HTTP/бота
- `docker/`, `docker-compose.yml` — деплой бота
- **`docs/SETUP.md`** — пошаговая инструкция запуска ⬅ начни отсюда
- `docs/DEFENSE.md` — материалы для защиты

## Quick start
```powershell
# 1) ML-сервис на хосте (GPU)
& "E:\dev\Python\313\Scripts\pip.exe" install -r ml_service\requirements.txt
$env:ML_DEVICE="cuda"; $env:ML_LLM_QUANT="int4"
& "E:\dev\Python\313\python.exe" -m uvicorn ml_service.server:app --host 0.0.0.0 --port 8000

# 2) Бот в Docker (в другом терминале)
Copy-Item .env.example .env       # вписать TELEGRAM_BOT_TOKEN
docker compose up --build -d bot
docker compose logs -f bot
```

Подробности, обучение, отладка → [docs/SETUP.md](docs/SETUP.md).
