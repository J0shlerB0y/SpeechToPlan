# Лёгкий образ ТОЛЬКО для Telegram-бота.
# ML (torch/whisper/transformers) сюда НЕ кладём — он крутится на хосте с GPU.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY bot/requirements.txt /app/bot/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/bot/requirements.txt

COPY bot     /app/bot
COPY shared  /app/shared

ENV PYTHONPATH=/app
CMD ["python", "-m", "bot.main"]
