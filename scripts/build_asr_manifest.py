"""Скачивает N примеров из HF-датасета и складывает их в локальную папку,
плюс пишет JSONL-манифест {audio, text} для scripts/eval_whisper.py.

Пример:
    python -m scripts.build_asr_manifest \
        --dataset bond005/sberdevices_golos_10h_crowd \
        --split test --n 30 \
        --out-dir .\llm\data\asr_samples \
        --manifest .\llm\data\asr_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bond005/sberdevices_golos_10h_crowd")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--text-field", default="transcription")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out-dir", default="./llm/data/asr_samples")
    p.add_argument("--manifest", default="./llm/data/asr_eval.jsonl")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")

    print(f"Скачиваем {args.dataset} [{args.split}], {args.n} примеров…")
    ds = load_dataset(args.dataset, args.config, split=args.split, token=hf_token, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    rows = []
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        wav = ex["audio"]["array"]
        sr  = ex["audio"]["sampling_rate"]
        text = ex.get(args.text_field) or ex.get("text") or ""
        if not text.strip():
            continue
        path = out_dir / f"sample_{i:03d}.wav"
        sf.write(path, wav, sr, subtype="PCM_16")
        rows.append({"audio": str(path).replace("\\", "/"), "text": text})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{args.n}] {path.name} :: {text[:60]!r}")

    Path(args.manifest).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\nГотово: {len(rows)} файлов → {out_dir}")
    print(f"Манифест → {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
