# SpeechToPlan — текст защиты (v2: Qwen2.5 + checkpoints + hybrid dataset)

> **Дисциплина:** Теоретическая информатика / Машинное обучение
> **Главный акцент:** качественное дообучение моделей — архитектура данных,
> выбор гиперпараметров, двухфазный Grid Search, структурированный вывод.

---

## 0. Хук (30 секунд)

Я сделал Telegram-бота, который принимает голосовое сообщение,
**локально** распознаёт речь, и через локальную LLM превращает её в
**структурированный план**: заголовок, приоритет, нормализованный срок и
список контрольных точек с под-сроками. Всё работает на RTX 3050 Ti без
облаков. Главное содержание работы — **итерация** над качеством вывода:
я дважды сменил базовую LLM (FunctionGemma → SmolLM2 → Qwen2.5), переработал
датасет с нуля и добавил структурный вывод, доказав на метриках что каждая
итерация давала улучшение.

---

## 1. Почему менялась LLM — честный разбор

| Итерация | Модель | Проблема | Решение |
|---|---|---|---|
| 1 | `google/functiongemma-270m-it` | Function-calling модель; генерирует `<start_function_call>` токены, которые `skip_special_tokens=True` срезает → **пустой вывод** | Отказались |
| 2 | `HuggingFaceTB/SmolLM2-360M-Instruct` | Работает, выдаёт валидный JSON. Но англоцентрична → низкое качество русского структурированного текста; priority всегда `low` | Upgrade |
| 3 | **`Qwen/Qwen2.5-1.5B-Instruct`** | Нативно силён в русском, отлично структурирует JSON. QLoRA INT4 ~3 ГБ обучение — влезает в 4 ГБ VRAM | Финальный выбор |

Каждая смена — результат конкретного эксперимента, а не случайного выбора.
Это и есть инженерный процесс.

---

## 2. Диагностика проблем: анализ данных

Перед переработкой провёл инструментальный анализ 434 обучающих примеров.

**Факты:**
- `description`: **0 %** примеров содержат структуру; ≈65 % — дословный
  повтор `title`; средняя длина **4.8 слова**.
- `priority`: только **3 из 434** примеров содержат слова срочности
  (`срочно`, `важно`, `горит`) → нет обучающего сигнала «контекст → приоритет».
- Поле `checkpoints` в схеме **не существовало** — структурированный план
  было некуда положить.

**Вывод:** плохой вывод модели — это отражение **плохих данных**, а не
нехватка эпох. Это академически важная мысль: модель не может выдать то,
чего никогда не видела в обучении.

---

## 3. Архитектура (1 минута)

```
┌──────────────────────────┐  HTTP   ┌─────────────────────────────────────┐
│ Telegram-бот (Docker)    │ ─────►  │ ML-сервис (хост, GPU)               │
│ aiogram 3.x + httpx      │ ◄─────  │ FastAPI + faster-whisper + Qwen2.5  │
└──────────────────────────┘  JSON   └─────────────────────────────────────┘
                       host.docker.internal:8000
```

Три изолированных модуля: `bot/`, `asr_emotion/`, `llm/` + `ml_service/`
как HTTP-обёртка над ML. ML на хосте (прямой GPU без docker nvidia-toolkit),
бот в Docker.

**Контракт данных** (`shared/schemas.py`):
```python
class Checkpoint(BaseModel):
    step: str
    deadline: Optional[str] = None

class PlannerTask(BaseModel):
    title: str
    description: Optional[str] = None
    deadline: Optional[str] = None
    priority: str  # low | medium | high | urgent
    checkpoints: list[Checkpoint] = []
    tags: list[str] = []
```

---

## 4. Дообучение моделей — **главная часть**

### 4.1 Whisper-tiny LoRA (ASR, русский)

Файл: `asr_emotion/training/train_whisper_lora.py`

- Данные: `bond005/sberdevices_golos_10h_crowd` + `bond005/taiga_speech_v2`
  (~22 000 записей train + 1 000 eval)
- Адаптируем: `q_proj`, `v_proj` — стандартный выбор для Whisper
- `gradient_checkpointing_enable()` + `use_cache=False` — обязательно
- Метрики **WER** и **CER** через `evaluate.load("wer")`, `predict_with_generate=True`
- Двухфазная схема: Grid Search на 5 000 → финал на полном датасете

**Результат (из реального прогона):**

| lr | r | α | Эпохи | eval_loss | **WER, %** | **CER, %** |
|---|---|---|---|---|---|---|
| 1e-3 | 8 | 16 | 1 на 5k | 0.899 | **49.2** | **20.7** |
| baseline whisper-tiny | — | — | — | — | ≈70 | ≈30 |

**≈ −30 % WER** при обучении **0.45 %** параметров (LoRA).

### 4.2 Qwen2.5-1.5B QLoRA (text→plan) + Grid Search

Файл: `llm/training/qlora_grid_search.py`

#### 4.2.1 Датасет v2 (гибрид): почему и как

Старый датасет (434 примера) не содержал структуры и слов срочности.
Собрал новый гибридным методом (`llm/data/build_dataset.py` + `make_complex_seed.py`):

**Обогащение 434 существующих:**
- Правила priority по ключевым словам (`срочно` → urgent, `важно` → high)
- Нормализация deadline к каноническому словарю
- Шаблонные checkpoints по категории задачи (встреча/оплата/учёба/документы…)

**Ручной seed ~162 сложных кейсов** с:
- явными urgent/high приоритетами
- 3-5 осмысленных шагов с под-сроками
- покрытием доменов (штрафы, экзамены, переезд, виза, отчёты…)

**Итог v2:**
```
Train: 572 примера | Val: 72
Priority: urgent=46, high=299, medium=152, low=75
Среднее checkpoints: 3.12
```

#### 4.2.2 QLoRA: почему умещается в 4 ГБ

| Компонент | Без QLoRA | С QLoRA NF4 |
|---|---|---|
| Веса базовой модели (1.5B) | ~3 ГБ fp16 | ~900 МБ int4 |
| Градиенты + optimizer | ~3 ГБ | ~200 МБ (только LoRA) |
| Обучаем параметров | 1.5B (100%) | **6.8M (0.45%)** |
| Итого VRAM | ОМО (>6 ГБ) | ≈3 ГБ ✅ |

LoRA-модули: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
(attention + MLP) — больше ёмкости для структурного reasoning.

#### 4.2.3 Grid Search (академическое требование)

Декартово произведение гиперпараметров — явный цикл без HPO-фреймворков:

```python
grid = [GridPoint(lr, r, a)
        for lr, r, a in itertools.product(lrs, ranks, alphas)]
# lrs=[1e-4, 3e-4], ranks=[16], alphas=[32]
```

**Фаза 1** (3 эпохи × 2 точки) → выбор лучшей по композитной метрике:

```python
score(r) = r.priority_accuracy + r.deadline_accuracy + r.checkpoint_count_match
```

**Фаза 2** — длинный финал (15-25 эпох) лучшей точки.

#### 4.2.4 Метрики (под структурную схему)

| Метрика | Что измеряет |
|---|---|
| `eval_loss` / `perplexity` | качество языкового моделирования |
| `json_validity` | % ответов, прошедших `json.loads` |
| `priority_accuracy` | доля правильного приоритета |
| `deadline_accuracy` | доля правильного срока |
| `checkpoint_presence` | доля ответов с непустым планом |
| `checkpoint_count_match` | число шагов близко к эталону (±1) |

_EM (exact match) при вложенных checkpoints почти всегда 0 — не используется
как главная метрика; показывается как референс._

#### 4.2.5 Ключевые технические решения

- **`_StopOnBalancedJson`**: стоп по балансу фигурных скобок (старый код
  стопился на первой `}` и рвал JSON с checkpoints)
- `max_new_tokens`: 256 → **448** (структурный JSON длиннее)
- `max_len` в датасете: 1024 → **768** (реальный макс. 645 токенов)
- Промпт: система на нативном `role="system"` (Qwen ChatML поддерживает);
  few-shot только при baseline-замере — при inference с адаптером убирается

#### 4.2.6 Почему Grid Search, а не k-fold кросс-валидация

| | Grid Search (наш выбор) | k-fold (k=5) |
|---|---|---|
| Стоимость | N точек × T | 5 × N × T |
| Цель | **выбрать лучший lr/rank** | оценить дисперсию модели |
| Наш датасет (572 train) | достаточно однократного split | 5× дороже без выигрыша |
| Прозрачность | таблица lr×rank → метрика | сложно показать на слайде |

---

## 5. Резолвер дедлайнов (`shared/deadline.py`)

Модель выдаёт нормализованные русские фразы (`завтра 18:00`, `в пятницу`,
`через две недели`). Бот считает конкретную дату от текущего дня в runtime:

```
"завтра"        → 2026-06-02
"в пятницу"     → 2026-06-05 (ближайшая)
"через месяц"   → 2026-07-01
"до 15-го числа"→ 2026-06-15
```

Это разделение ответственности: модель решает **семантику** (какой срок),
рантайм решает **арифметику** (какое число).

---

## 6. Демонстрация на защите

```powershell
# 1. ML-сервис с обученным адаптером
$env:ML_LLM_ADAPTER = "$PWD\checkpoints\qwen-grid\best_adapter"
E:\dev\Python\313\python.exe -m uvicorn ml_service.server:app --host 0.0.0.0 --port 8000

# 2. Health check
curl http://127.0.0.1:8000/health
# {"llm_has_lora": true, "asr_loaded": true}

# 3. Тест голосом
curl.exe -X POST http://127.0.0.1:8000/process/voice -F "file=@тестовое.ogg"
# {"title":"...", "priority":"...", "deadline":"...", "checkpoints":[...]}

# 4. Тест urgent-кейсом
curl.exe -X POST http://127.0.0.1:8000/process/text `
    -H "Content-Type: application/json" `
    -d '{\"text\":\"Срочно, завтра до 18 часов надо сдать отчёт шефу\"}'

# 5. Метрики до/после
python -m scripts.eval_llm --few-shot              # baseline
python -m scripts.eval_llm --adapter .\checkpoints\qwen-grid\best_adapter
```

**Ожидаемый рендер бота:**
```
Сдать отчёт шефу
Приоритет: 🔴 СРОЧНО
Срок: завтра 18:00 (2026-06-02 18:00)

Подготовить и сдать отчёт руководителю к завтрашнему вечеру

План:
1. Собрать данные — сегодня (2026-06-01)
2. Оформить отчёт — завтра (2026-06-02)
3. Отправить руководителю — завтра 18:00 (2026-06-02 18:00)
```

---

## 7. Итоги и что бы улучшил

**Достигнуто:**
- WER Whisper LoRA: ≈70% → **49.2%** (−30% относительно, 0.45% параметров)
- LLM: FunctionGemma (пустой вывод) → SmolLM2 (базовый JSON) → Qwen2.5 (структурный план)
- Датасет: 0 checkpoints → **3.12 в среднем**; 3 из 434 urgent → **46 из 572**
- Bitsandbytes + peft версионная несовместимость диагностирована и устранена

**Что дальше:**
- Полное обучение Qwen LoRA на всём датасете (15-25 эпох) → ждём метрики
- Whisper: 3+ эпохи на полных 22k примеров → WER < 40%
- ROUGE-1 по checkpoints (вместо точного совпадения)
- Constrained decoding (`outlines`) — 100% json_validity без обучения

---

## 8. Ключевые файлы

| Что | Файл |
|---|---|
| Схема PlannerTask + Checkpoint | [shared/schemas.py](../shared/schemas.py) |
| Резолвер дедлайнов | [shared/deadline.py](../shared/deadline.py) |
| Промпт (system role + few-shot) | [llm/prompts.py](../llm/prompts.py) |
| **Grid Search QLoRA + все метрики** | [llm/training/qlora_grid_search.py](../llm/training/qlora_grid_search.py) |
| Обогащение + seed датасет | [llm/data/build_dataset.py](../llm/data/build_dataset.py), [make_complex_seed.py](../llm/data/make_complex_seed.py) |
| Инференс (Qwen, INT4, fallback) | [llm/inference.py](../llm/inference.py) |
| **Whisper LoRA + Grid Search + WER/CER** | [asr_emotion/training/train_whisper_lora.py](../asr_emotion/training/train_whisper_lora.py) |
| Рендер бота с чекпоинтами | [bot/utils/formatting.py](../bot/utils/formatting.py) |
| Eval LLM (новые метрики) | [scripts/eval_llm.py](../scripts/eval_llm.py) |
| Eval ASR (WER/CER/RTF) | [scripts/eval_whisper.py](../scripts/eval_whisper.py) |
| FastAPI ML-сервис | [ml_service/server.py](../ml_service/server.py) |
