# CUDA 12.1 + cuDNN 8 — совместимо с torch 2.4.1 и bitsandbytes 0.44.1.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchaudio==2.4.1 && \
    pip install -r /app/requirements.txt

COPY bot         /app/bot
COPY asr_emotion /app/asr_emotion
COPY llm         /app/llm
COPY shared      /app/shared

ENV PYTHONPATH=/app

CMD ["python", "-m", "bot.main"]
