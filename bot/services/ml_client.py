from __future__ import annotations

import logging
from pathlib import Path

import httpx

from shared.schemas import PlannerTask

log = logging.getLogger(__name__)


class MLClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        r = await self._client.get("/health")
        r.raise_for_status()
        return r.json()

    async def process_text(self, text: str) -> PlannerTask:
        r = await self._client.post("/process/text", json={"text": text})
        r.raise_for_status()
        return PlannerTask.model_validate(r.json())

    async def process_voice(self, audio_path: Path) -> PlannerTask:
        with audio_path.open("rb") as fh:
            files = {"file": (audio_path.name, fh, "application/octet-stream")}
            r = await self._client.post("/process/voice", files=files)
        r.raise_for_status()
        return PlannerTask.model_validate(r.json())
