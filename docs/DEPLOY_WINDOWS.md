# Деплой на Windows с GPU (RTX 3050 Ti)

> Запускаем контейнер CUDA внутри Docker Desktop, GPU пробрасываем через WSL2.
> Никаких драйверов внутри контейнера ставить не нужно — только в хосте.

## Шаг 1. Настроить хост (один раз)
1. Установить актуальный **NVIDIA Game Ready / Studio Driver** (≥ 535) — он несёт
   `nvidia-smi`, который Docker Desktop пробрасывает в WSL2.
2. Установить **Docker Desktop ≥ 4.27** и в `Settings → General` включить
   **Use the WSL 2 based engine**, в `Settings → Resources → WSL Integration` —
   включить интеграцию с дистрибутивом Ubuntu.
3. В PowerShell проверить: `wsl -d Ubuntu nvidia-smi` должен показать вашу 3050 Ti.
   Если показывает — Docker увидит GPU автоматически (CDI hook уже встроен).

## Шаг 2. Подготовить конфиг и собрать образ
Из корня проекта:
```powershell
Copy-Item .env.example .env
notepad .env                   # вписать TELEGRAM_BOT_TOKEN
docker compose build           # ~10 минут на первый раз (torch + CUDA-зависимости)
```

## Шаг 3. Запуск
```powershell
docker compose up -d
docker compose logs -f speechtoplan
```
В логах должно быть `Модели в VRAM. Стартуем polling.` — после этого бот в Telegram
отвечает на `/start`. Веса HuggingFace кэшируются в volume `hf_cache` — повторный
старт уже без скачивания.

---

### Подсказки по диагностике

| Симптом | Причина | Что сделать |
|---|---|---|
| `could not select device driver "nvidia"` | Docker не видит GPU из WSL2 | Перезапустить Docker Desktop и проверить `wsl -d Ubuntu nvidia-smi` |
| `CUDA out of memory` при старте | Грузится всё одновременно | В `.env` поставить `LLM_QUANT=int4` и `ASR_MODEL_SIZE=tiny` |
| `bitsandbytes` не находит CUDA | Несовместимость драйвера/CUDA | Использовать базовый образ `nvidia/cuda:12.1.1-cudnn8-runtime` (по умолчанию) |
| Долгая первая загрузка | Скачивание весов в volume | Это разовое; см. `hf_cache` volume |
