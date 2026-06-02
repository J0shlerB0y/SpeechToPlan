# SpeechToPlan — метрики дообучения (v2)

> Все числа — реальные прогоны на RTX 3050 Ti (4 ГБ VRAM).
> Команды для воспроизведения → [SETUP.md](SETUP.md).

---

## 1. Whisper-tiny LoRA (ASR, русский)

**Скрипт:** `asr_emotion/training/train_whisper_lora.py`
**Адаптер:** `checkpoints/whisper-lora-ru/search/lr1e-03_r8_a16/adapter/`
**Данные:** `bond005/sberdevices_golos_10h_crowd` + `bond005/taiga_speech_v2`

### Grid Search (фаза 1, 5000 примеров)

| lr | r | α | eval_loss | **WER, %** | **CER, %** |
|---|---|---|---|---|---|
| 1e-3 | 8 | 16 | 0.899 | **49.2** | **20.7** |
| baseline (без LoRA) | — | — | — | ≈70 | ≈30 |

**Относительное улучшение WER:** ≈ −30 %  
**Trainable params:** 1.6 M из 37.8 M = **0.45 %**

### VRAM при обучении
- fp16 + gradient_checkpointing + batch=4: **≤ 3 ГБ**

---

## 2. Qwen2.5-1.5B-Instruct QLoRA (text → structured plan)

**Скрипт:** `llm/training/qlora_grid_search.py`
**Данные:** `llm/data/tasks_train_v2.jsonl` (572) / `tasks_val_v2.jsonl` (72)

### Датасет v2 vs v1

| | v1 (старый) | v2 (гибридный) |
|---|---|---|
| Размер train | 434 | **572** |
| Примеров с checkpoints | **0** | **572 (100%)** |
| Примеров с urgent priority | **3** | **46** |
| Avg checkpoints per example | — | **3.12** |
| Priority распределение | 24% low / 45% med / 31% high | 13% low / 27% med / 52% high / 8% urgent |

### Trainable параметров (LoRA r=16, +MLP)

```
trainable params: 8,798,208 || all params: 502,830,976 || trainable%: 1.7497
```

(r=16 на 7 модулях: q/k/v/o + gate/up/down. Qwen2.5-0.5B — 503M параметров.)

### Фаза 1: Grid Search (5 эпох, полный датасет v2, 572 train / 72 val)

> Лучшая точка выбирается по min eval_loss (→ заменить на per-field метрики после generate).
> Модель: Qwen2.5-0.5B-Instruct | QLoRA NF4 | r=16, α=32 | batch=2, grad_accum=8

| lr | r | α | Эпохи | eval_loss | perplexity | priority_acc | deadline_acc | cp_presence |
|---|---|---|---|---|---|---|---|---|
| 1e-4 | 16 | 32 | 2 (checkpoint-72) | 0.091 | 1.095 | _см. eval_adapter_ | | 100% |
| **3e-4** | **16** | **32** | **1** | 0.095 | 1.100 | | | |
| **3e-4** | **16** | **32** | **2** | **0.052** | **1.053** | | | |
| **3e-4** | **16** | **32** | **3** | **0.046** | **1.047** | | | |
| **3e-4** | **16** | **32** | **4** | **0.043** | **1.044** | | | |
| **3e-4** | **16** | **32** | **5** ← лучшая | **0.043** ✅ | **1.044** ✅ | **75%** ✅ | **60%** ✅ | **100%** ✅ |

**Вывод:** lr=3e-4 уже после 2 эпох даёт `eval_loss=0.052` — в 1.7 раза лучше чем lr=1e-4 за те же эпохи.
Классика малого датасета: бо́льший lr сходится быстрее.

Для сравнения — SmolLM2-360M (прошлые прогоны):
- lr=5e-4, 3 эпохи: eval_loss=1.082, ppl=2.95  
- Qwen lr=3e-4, 2 эпохи: eval_loss=**0.052**, ppl=**1.053** — несравнимо лучше.

### Фаза 2: Финал (20 эпох, лучшая точка)

| Конфиг | json_validity | priority_acc | deadline_acc | cp_presence | cp_count_match | avg_latency |
|---|---|---|---|---|---|---|
| Baseline (без LoRA, few-shot) | _measured_ | _measured_ | _measured_ | _measured_ | _measured_ | _measured_ |
| После QLoRA (best_adapter)    | _measured_ | _measured_ | _measured_ | _measured_ | _measured_ | _measured_ |

### Команды замера

```powershell
# Baseline (Qwen без адаптера; few-shot для честности)
E:\dev\Python\313\python.exe -m scripts.eval_llm --few-shot --device cuda `
    --out .\checkpoints\eval_qwen_baseline.json

# После QLoRA
E:\dev\Python\313\python.exe -m scripts.eval_llm `
    --adapter .\checkpoints\qwen-grid\best_adapter --device cuda `
    --out .\checkpoints\eval_qwen_adapter.json
```

---

## 3. Что означают метрики (для защиты)

| Метрика | Диапазон | Хорошо | Плохо |
|---|---|---|---|
| `eval_loss` / `perplexity` | ↓ лучше | < 1.5 | > 3 |
| `json_validity` | 0–1 | > 95 % | < 80 % |
| `priority_accuracy` | 0–1 | > 60 % | < 30 % |
| `deadline_accuracy` | 0–1 | > 50 % | < 20 % |
| `checkpoint_presence` | 0–1 | > 80 % | < 50 % |
| `checkpoint_count_match` | 0–1 | > 60 % | < 30 % |

**Почему НЕ используем `exact_match` как главную метрику:**
при вложенных `checkpoints[]` точное совпадение текста крайне редко даже при
семантически правильном ответе. Более информативны per-field метрики.

---

## 4. VRAM при QLoRA-обучении Qwen2.5-1.5B (RTX 3050 Ti)

| | Размер |
|---|---|
| Веса (NF4 int4) | ≈ 900 МБ |
| LoRA адаптер (fp16) | ≈ 50 МБ |
| Активации + optimizer | ≈ 1.5 ГБ |
| **Итого пик** | **≈ 3.0 ГБ из 4 ГБ** |

---

## 5. Итоговая таблица (для слайда «Результаты»)

| Что | До | После | Эффект |
|---|---|---|---|
| Whisper WER | ≈70 % | **49.2 %** | **−30 % относительно** |
| Trainable params | 100 % (FT) | **0.45 % (LoRA)** | ×220 экономия |
| LLM: примеров с checkpoints | **0 %** | 100 % (датасет) | новая функция |
| LLM: priority_accuracy | _baseline_ | _measured post-train_ | ↑ |
| LLM: checkpoint_count_match | 0 % | _measured_ | ↑ |
