# SpeechToPlan

Локальный Telegram-ассистент-планировщик: голос/текст → структурированная JSON-задача.

```
┌──────────┐   text    ┌────────────────┐               ┌────────────┐
│ Telegram │──────────►│ /bot (aiogram) │──────────────►│   /llm     │
│  voice   │   .ogg    │  оркестрация   │   text+emo    │ FunctionGemma│
└──────────┘           └──────┬─────────┘               └─────┬──────┘
                              │ audio path                    │ JSON
                              ▼                               ▼
                     ┌────────────────────┐         ┌──────────────────┐
                     │   /asr_emotion     │         │   PlannerTask    │
                     │ Whisper ║ wav2vec2 │         │ (Pydantic)       │
                     └────────────────────┘         └──────────────────┘
```

## Структура
- `bot/` — aiogram-бот, точка входа `python -m bot.main`
- `asr_emotion/` — параллельный ASR (faster-whisper) + SER (wav2vec2)
- `llm/` — FunctionGemma-270m-it: инференс и QLoRA-обучение с Grid Search
- `shared/` — Pydantic-схемы, общие для всех модулей
- `docker/`, `docker-compose.yml` — деплой
- `docs/DEFENSE.md` — материалы для защиты
- `docs/DEPLOY_WINDOWS.md` — деплой на Windows + RTX 3050 Ti

## Быстрый старт (Docker)
```powershell
Copy-Item .env.example .env       # вписать TELEGRAM_BOT_TOKEN
docker compose up --build -d
docker compose logs -f speechtoplan
```

## Быстрый старт (без Docker)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:TELEGRAM_BOT_TOKEN = "..."
python -m bot.main
```

## Обучение моделей (RTX 3050 Ti, 4 ГБ)

### Whisper-tiny LoRA + WER
```powershell
python -m asr_emotion.training.train_whisper_lora `
    --dataset mozilla-foundation/common_voice_17_0 `
    --lang ru --output .\checkpoints\whisper-tiny-ru-lora
```

### FunctionGemma QLoRA + Grid Search
```powershell
python -m llm.training.qlora_grid_search `
    --train-file .\llm\data\tasks_train.jsonl `
    --val-file .\llm\data\tasks_val.jsonl `
    --output .\checkpoints\gemma-grid
```
Результат: `checkpoints/gemma-grid/grid_summary.json` (таблица результатов)
и `checkpoints/gemma-grid/best_adapter/` (лучший LoRA-адаптер). Подключается
переменной окружения `LLM_ADAPTER_PATH`.
