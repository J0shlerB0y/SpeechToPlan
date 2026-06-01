# SpeechToPlan — полная инструкция по запуску

> **Архитектура (после рефакторинга от 2026-05-17)**
>
> ```
>   ┌──────────────────────────┐  HTTP   ┌──────────────────────────────────┐
>   │ Telegram-бот (Docker)    │ ─────►  │ ML-сервис (хост, GPU)            │
>   │ aiogram 3.x + httpx      │ ◄─────  │ FastAPI + faster-whisper +       │
>   │                          │ JSON    │ wav2vec2 SER + FunctionGemma     │
>   └──────────────────────────┘         └──────────────────────────────────┘
>                          host.docker.internal:8000
> ```
>
> **Почему так.** На Windows torch+CUDA + bitsandbytes в Docker — больно
> (требует настроенный `nvidia-container-toolkit` в WSL2, который часто
> не пробрасывает `libnvidia-ml.so.1`). Поэтому ML-стек крутится на хосте
> напрямую с GPU, а в Docker остаётся только лёгкий Telegram-бот.

---

## 0. Системные требования

| Что | Версия | Зачем |
|---|---|---|
| Windows 11 (или 10) | — | хост |
| NVIDIA driver | ≥ 535 | для CUDA на хосте |
| Python | 3.11 — 3.13 (у меня 3.13: `E:\dev\Python\313\python.exe`) | ML-сервис |
| Docker Desktop | ≥ 4.27 | для бота |
| Telegram Bot Token | через @BotFather | связь с пользователями |
| HuggingFace token | _опционально_, для gated моделей (Gemma/FunctionGemma) | загрузка весов |

---

## 1. ML-сервис на хосте

### 1.1 torch с поддержкой CUDA
В системном Python сейчас стоит `torch 2.12.0+cpu` — это **CPU-only**, GPU работать не будет.
Переустанови на CUDA-сборку (для CUDA 12.x на RTX 3050 Ti берём `cu124`):

```powershell
& "E:\dev\Python\313\Scripts\pip.exe" uninstall -y torch torchaudio torchvision
& "E:\dev\Python\313\Scripts\pip.exe" install --index-url https://download.pytorch.org/whl/cu124 `
    torch torchaudio
& "E:\dev\Python\313\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# должно напечатать: True 12.4
```

> Если по какой-то причине не хочется ставить CUDA-torch — можно крутить всё на CPU
> (медленнее в 5-10 раз, но работает). Тогда экспортируй `ML_DEVICE=cpu` и `ML_LLM_QUANT=none`.

### 1.2 ML-зависимости
```powershell
& "E:\dev\Python\313\Scripts\pip.exe" install -r ml_service\requirements.txt
```

Это поставит: faster-whisper, transformers, peft, accelerate, soundfile, PyAV, fastapi, uvicorn.
**`bitsandbytes`** в этом requirements нет специально — он капризен на Windows.
Если будешь использовать `ML_LLM_QUANT=int4|int8` (нужен для FunctionGemma на 4 ГБ VRAM),
поставь его отдельно (он работает только под CUDA-torch):

```powershell
& "E:\dev\Python\313\Scripts\pip.exe" install bitsandbytes
```

### 1.3 (опционально) HuggingFace токен для FunctionGemma
`google/functiongemma-270m-it` — **gated repo**. Зайди на её страницу, нажми
*Acknowledge license*, потом сгенерируй токен в settings/tokens и логинься:

```powershell
& "E:\dev\Python\313\Scripts\pip.exe" install huggingface_hub
& "E:\dev\Python\313\Scripts\huggingface-cli.exe" login
# вставить токен hf_***
```

Если токен оформлять не хочется — для разработки/защиты подойдёт публичная мини-LLM:
поставь в .env (см. ниже) `ML_LLM_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct`.

### 1.4 Запуск ML-сервиса
Из корня проекта:

```powershell
$env:ML_DEVICE       = "cuda"        # или "cpu"
$env:ML_LLM_QUANT    = "int4"        # int4 / int8 / none
$env:ML_LLM_MODEL    = "google/functiongemma-270m-it"   # или SmolLM2-360M-Instruct
$env:ML_ASR_SIZE     = "tiny"
$env:ML_EMOTION_MODEL= "superb/wav2vec2-base-superb-er"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

& "E:\dev\Python\313\python.exe" -m uvicorn ml_service.server:app --host 0.0.0.0 --port 8000
```

После загрузки моделей (10-90 с в зависимости от того, скачаны ли веса):

```powershell
curl http://127.0.0.1:8000/health
# {"status":"ok","asr":true,"llm":true}
```

> `--host 0.0.0.0` обязателен, иначе Docker-бот не достучится через `host.docker.internal`.

---

## 2. Telegram-бот в Docker

### 2.1 .env
```powershell
Copy-Item .env.example .env
notepad .env
# вписать TELEGRAM_BOT_TOKEN=...
```

### 2.2 Сборка и запуск
```powershell
docker compose build bot           # быстро: ~30 с (нет torch)
docker compose up -d bot
docker compose logs -f bot
```

Ожидаемые строки:
```
Подключаемся к ML-сервису: http://host.docker.internal:8000
ML-сервис отвечает: {'status': 'ok', 'asr': True, 'llm': True}
Бот готов. Стартуем polling.
```

Открой бота в Telegram, отправь голосовое — должна прилететь JSON-задача.

---

## 3. Smoke-тесты (без Telegram)

### 3.1 Локальный прогон pipeline целиком
```powershell
& "E:\dev\Python\313\python.exe" -m scripts.test_pipeline ".\тестовое.ogg" --device cuda --quant int4
```

### 3.2 Дёрнуть HTTP-сервис голосом
```powershell
curl.exe -X POST "http://127.0.0.1:8000/process/voice" -F "file=@тестовое.ogg"
```

### 3.3 Дёрнуть HTTP-сервис текстом
```powershell
curl.exe -X POST "http://127.0.0.1:8000/process/text" `
    -H "Content-Type: application/json" `
    -d '{\"text\":\"Завтра в 9 утра встреча с заказчиком\"}'
```

### 3.4 Проверить сеть из Docker → хост
```powershell
docker run --rm --add-host=host.docker.internal:host-gateway speechtoplan-bot:latest `
    python -c "import httpx; print(httpx.get('http://host.docker.internal:8000/health').json())"
```

---

## 4. Обучение моделей (отдельные запуски, не нужны для работы бота)

### 4.0 Подготовка датасета v2
```powershell
# Сгенерировать сложные кейсы (~162 примера)
& "E:\dev\Python\313\python.exe" -m llm.data.make_complex_seed

# Обогатить существующие 434 + добавить seed → v2
& "E:\dev\Python\313\python.exe" -m llm.data.build_dataset
# Результат: llm/data/tasks_train_v2.jsonl (572) + tasks_val_v2.jsonl (72)
```

### 4.1 QLoRA Grid Search + финальное обучение LLM (Qwen2.5)
```powershell
# Smoke-тест пайплайна (16 примеров, 1 эпоха, ~10 мин)
& "E:\dev\Python\313\python.exe" -m llm.training.qlora_grid_search `
    --base-model Qwen/Qwen2.5-0.5B-Instruct `
    --quant nf4 --device cuda --epochs 1 --final-epochs 0 `
    --train-limit 16 --val-limit 6 --lrs 3e-4 --ranks 16 --alphas 32

# Полный прогон: 2 точки grid (5 эпох) + финал 15 эпох (~2-3 ч)
& "E:\dev\Python\313\python.exe" -m llm.training.qlora_grid_search `
    --base-model Qwen/Qwen2.5-0.5B-Instruct `
    --quant nf4 --device cuda `
    --epochs 5 --final-epochs 15 `
    --batch 2 --grad-accum 8 `
    --lrs 1e-4 3e-4 --ranks 16 --alphas 32 `
    --output .\checkpoints\qwen05-grid

# Или на 1.5B (лучше качество, но медленнее)
& "E:\dev\Python\313\python.exe" -m llm.training.qlora_grid_search `
    --base-model Qwen/Qwen2.5-1.5B-Instruct `
    --output .\checkpoints\qwen-grid
# ... те же параметры
```

Результат: `checkpoints/qwen05-grid/best_adapter/` (или `qwen-grid/`)  
Таблица Grid Search: `checkpoints/qwen05-grid/grid_summary.json`

### 4.2 LoRA-дообучение Whisper-tiny с метрикой WER
```powershell
& "E:\dev\Python\313\python.exe" -m asr_emotion.training.train_whisper_lora `
    --base-model openai/whisper-tiny `
    --dataset mozilla-foundation/common_voice_17_0 `
    --lang ru `
    --output .\checkpoints\whisper-tiny-ru-lora
```

После завершения — адаптер в `.\checkpoints\qwen05-grid\best_adapter`.
Подключи его к ML-сервису:
```powershell
$env:ML_LLM_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
$env:ML_LLM_ADAPTER = "$PWD\checkpoints\qwen05-grid\best_adapter"
& "E:\dev\Python\313\python.exe" -m uvicorn ml_service.server:app --host 0.0.0.0 --port 8000
```

### 4.3 Eval «до/после»
```powershell
# Baseline (Qwen без адаптера, с few-shot)
& "E:\dev\Python\313\python.exe" -m scripts.eval_llm --few-shot --device cuda `
    --model Qwen/Qwen2.5-0.5B-Instruct `
    --out .\checkpoints\eval_qwen_baseline.json

# После обучения
& "E:\dev\Python\313\python.exe" -m scripts.eval_llm --device cuda `
    --model Qwen/Qwen2.5-0.5B-Instruct `
    --adapter .\checkpoints\qwen05-grid\best_adapter `
    --out .\checkpoints\eval_qwen_adapter.json
```

---

## 5. Что куда смотрит

| Что хочешь поменять | Файл |
|---|---|
| Модель ASR (whisper size, язык) | env-перем. `ML_ASR_SIZE`, `ML_ASR_COMPUTE` |
| Модель SER | env-перем. `ML_EMOTION_MODEL` |
| Модель LLM | env-перем. `ML_LLM_MODEL` + `ML_LLM_QUANT` |
| LoRA-адаптер | env-перем. `ML_LLM_ADAPTER` |
| Подключение бота к ML | `bot/config.py`, env `ML_SERVICE_URL` |
| Промпт LLM, few-shot | [llm/prompts.py](../llm/prompts.py) |
| Контракт JSON | [shared/schemas.py](../shared/schemas.py) (`PlannerTask`) |
| Архитектура / обоснования | [docs/DEFENSE.md](DEFENSE.md) |

---

## 6. Известные проблемы и их решения

| Симптом | Причина | Лечение |
|---|---|---|
| `torch.cuda.is_available() == False` | стоит `torch+cpu` | переустановить на `cu124` (см. 1.1) |
| `operator torchvision::nms does not exist` | `torchvision` несовместим с твоим torch | `pip uninstall -y torchvision` (для аудио не нужен) |
| `Cannot access gated repo` для FunctionGemma | нет HF-токена / не принят license | см. 1.3 либо переключи `ML_LLM_MODEL` на `HuggingFaceTB/SmolLM2-360M-Instruct` |
| `bitsandbytes: cannot find CUDA` | библиотека требует CUDA-torch | поставь CUDA-torch (1.1), потом `pip install bitsandbytes` |
| Бот в Docker не достучится до ML | сервис слушает только `127.0.0.1` | запускай uvicorn с `--host 0.0.0.0` |
| `host.docker.internal` не резолвится на Linux | нужно добавить host alias | уже в `docker-compose.yml` (`extra_hosts: host.docker.internal:host-gateway`) |
| `ffmpeg: not found` при импорте audio | в системном PATH нет ffmpeg | проект использует PyAV — `ffmpeg` не нужен; убедись что `av` поставлен |
| GPU из Docker не видится (`libnvidia-ml.so.1`) | нет `nvidia-container-toolkit` в WSL2 | **именно поэтому ML на хосте, а не в Docker** |

---

## 7. Что было проверено end-to-end

На железе пользователя (Windows 11, RTX 3050 Ti 4 ГБ, Python 3.13 на `E:\dev\Python\313`):

1. ML-сервис на CPU + SmolLM2-360M-Instruct: запускается за ~70 с, `/health` → ok, `/process/voice` на `тестовое.ogg` возвращает валидный `PlannerTask` JSON.
2. Bot Docker-образ собирается за ~30 с (без torch).
3. Из Docker-бот-контейнера: `httpx.get('http://host.docker.internal:8000/health')` → 200 OK.
4. Из Docker-бот-контейнера: `POST /process/voice` с тем же `.ogg` → валидный JSON.

Качество ответа SmolLM2 ожидаемо посредственное (модель 360M);
с FunctionGemma после QLoRA Grid Search будет лучше — её можно подключить
после оформления HF-токена.
