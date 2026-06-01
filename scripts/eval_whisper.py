"""Оценка Whisper (с/без LoRA адаптера) на манифесте.

Манифест — JSONL:
    {"audio": "C:/path/to/a.ogg", "text": "эталонная транскрипция"}

Метрики:
  * WER, CER (через evaluate.load)
  * RTF (real-time factor = inference_time / audio_duration)
  * avg_latency_s
  * total_audio_s

Запуск:
    # faster-whisper baseline
    python -m scripts.eval_whisper --manifest .\llm\data\asr_eval.jsonl --backend faster

    # transformers + LoRA адаптер
    python -m scripts.eval_whisper --manifest .\llm\data\asr_eval.jsonl \
        --backend transformers \
        --adapter .\checkpoints\whisper-lora-ru\search\lr1e-03_r8_a16\adapter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_manifest(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--backend", choices=["faster", "transformers"], default="faster")
    p.add_argument("--size", default="tiny")
    p.add_argument("--compute", default="int8")
    p.add_argument("--adapter", default=None)
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument("--out", default="./checkpoints/asr_eval_report.json")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows = _load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    print(f"ASR eval: {len(rows)} файлов | backend={args.backend} adapter={args.adapter}")

    if args.backend == "transformers" or args.adapter:
        from asr_emotion.asr import WhisperASRTransformers
        asr = WhisperASRTransformers.load(
            size=args.size, device=args.device,
            compute_type=args.compute, adapter_path=args.adapter,
        )
    else:
        from asr_emotion.asr import WhisperASR
        asr = WhisperASR.load(size=args.size, device=args.device, compute_type=args.compute)

    try:
        import evaluate
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")
        have_eval = True
    except Exception:
        have_eval = False
        print("WARNING: evaluate.load('wer') не сработал, посчитаем простую WER вручную")

    preds: list[str] = []
    refs: list[str] = []
    total_audio = 0.0
    total_time = 0.0
    samples = []

    for i, ex in enumerate(rows, 1):
        t0 = time.perf_counter()
        res = asr.transcribe(ex["audio"])
        dt = time.perf_counter() - t0
        total_time += dt
        total_audio += res.duration_sec or 0.0

        preds.append(res.text)
        refs.append(ex["text"])
        samples.append({
            "audio": ex["audio"],
            "ref": ex["text"],
            "pred": res.text,
            "duration_s": res.duration_sec,
            "latency_s": round(dt, 3),
        })
        print(f"[{i}/{len(rows)}] {dt:.1f}s | ref={ex['text'][:60]!r} | pred={res.text[:60]!r}")

    if have_eval:
        wer = 100 * wer_metric.compute(predictions=preds, references=refs)
        cer = 100 * cer_metric.compute(predictions=preds, references=refs)
    else:
        # дешёвый word-level WER без evaluate
        from difflib import SequenceMatcher
        def _simple_wer(p, r):
            pw, rw = p.split(), r.split()
            if not rw:
                return 0.0
            sm = SequenceMatcher(a=rw, b=pw)
            return (1 - sm.ratio()) * 100
        wer = sum(_simple_wer(p, r) for p, r in zip(preds, refs)) / len(preds)
        cer = 0.0

    rtf = total_time / total_audio if total_audio else 0.0

    report = {
        "backend": args.backend,
        "size": args.size,
        "adapter": args.adapter,
        "n_examples": len(rows),
        "wer_percent": round(wer, 2),
        "cer_percent": round(cer, 2),
        "rtf": round(rtf, 3),
        "avg_latency_s": round(total_time / len(rows), 3) if rows else 0.0,
        "total_audio_s": round(total_audio, 1),
        "total_time_s": round(total_time, 1),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"report": report, "samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ASR METRICS ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nДетали → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
