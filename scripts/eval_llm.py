r"""Оценка LLM (с адаптером или без) на val.jsonl под структурную схему.

Метрики:
  * json_validity         — % ответов, прошедших json.loads
  * exact_match           — % полных совпадений (при checkpoints почти всегда 0)
  * priority_accuracy     — точность приоритета
  * deadline_accuracy     — точность срока
  * checkpoint_presence   — доля ответов с непустым планом
  * checkpoint_count_match— доля, где число шагов близко к эталону (±1)
  * avg_checkpoints, avg_latency_s

Запуск:
    # baseline (без адаптера, few-shot для честности)
    python -m scripts.eval_llm --few-shot

    # с адаптером
    python -m scripts.eval_llm --adapter .\checkpoints\qwen-grid\best_adapter
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_val(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="./llm/data/tasks_val_v2.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--quant", default="int4", choices=["int4", "int8", "none"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--few-shot", action="store_true",
                   help="few-shot подсказки (для честного baseline без адаптера)")
    p.add_argument("--limit", type=int, default=0,
                   help="0 = весь val, иначе N первых")
    p.add_argument("--out", default="./checkpoints/eval_report.json")
    args = p.parse_args()

    from llm.inference import GemmaPlannerLLM

    rows = _load_val(args.val)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Validation: {len(rows)} примеров | adapter={args.adapter} | few_shot={args.few_shot}")
    print("Грузим модель…")

    llm = GemmaPlannerLLM.load(
        model_path=args.model, quant=args.quant, device=args.device,
        adapter_path=args.adapter,
    )

    valid = em = 0
    field_hits = {k: 0 for k in ("title", "description", "deadline", "priority")}
    field_total = {k: 0 for k in field_hits}
    cp_present = cp_count_match = cp_gold_total = 0
    total_pred_cp = 0
    total_chars = 0
    total_time = 0.0
    samples = []

    for i, ex in enumerate(rows, 1):
        t0 = time.perf_counter()
        completion = llm.generate_raw(ex["input"], few_shot=args.few_shot)
        dt = time.perf_counter() - t0
        total_time += dt
        total_chars += len(completion)

        m = JSON_RE.search(completion or "")
        pred = None
        if m:
            try:
                pred = json.loads(m.group(0))
                valid += 1
            except json.JSONDecodeError:
                pred = None

        if pred and pred == ex["output"]:
            em += 1

        if pred:
            for k in field_hits:
                if k in ex["output"]:
                    field_total[k] += 1
                    if pred.get(k) == ex["output"].get(k):
                        field_hits[k] += 1
            # checkpoints
            pred_cp = pred.get("checkpoints") or []
            gold_cp = ex["output"].get("checkpoints") or []
            if isinstance(pred_cp, list) and len(pred_cp) > 0:
                cp_present += 1
                total_pred_cp += len(pred_cp)
            if gold_cp:
                cp_gold_total += 1
                if isinstance(pred_cp, list) and abs(len(pred_cp) - len(gold_cp)) <= 1:
                    cp_count_match += 1

        samples.append({
            "input": ex["input"],
            "gold": ex["output"],
            "pred_raw": completion,
            "pred_parsed": pred,
            "latency_s": round(dt, 3),
        })

        if i <= 3 or i % 5 == 0:
            print(f"[{i}/{len(rows)}] {dt:.1f}s | "
                  f"valid={valid} prio_ok={field_hits['priority']} cp={cp_present} | "
                  f":: {completion[:80]!r}")

    n = len(rows)
    report = {
        "model": args.model,
        "adapter": args.adapter,
        "few_shot": args.few_shot,
        "n_examples": n,
        "json_validity": valid / n if n else 0.0,
        "exact_match": em / n if n else 0.0,
        "priority_accuracy": field_hits["priority"] / field_total["priority"] if field_total["priority"] else 0.0,
        "deadline_accuracy": field_hits["deadline"] / field_total["deadline"] if field_total["deadline"] else 0.0,
        "title_accuracy": field_hits["title"] / field_total["title"] if field_total["title"] else 0.0,
        "description_accuracy": field_hits["description"] / field_total["description"] if field_total["description"] else 0.0,
        "checkpoint_presence": cp_present / n if n else 0.0,
        "checkpoint_count_match": cp_count_match / cp_gold_total if cp_gold_total else 0.0,
        "avg_checkpoints": total_pred_cp / cp_present if cp_present else 0.0,
        "avg_output_chars": total_chars / n if n else 0.0,
        "avg_latency_s": total_time / n if n else 0.0,
        "total_time_s": total_time,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "samples": samples},
                              ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n=== METRICS ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nДетали → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
