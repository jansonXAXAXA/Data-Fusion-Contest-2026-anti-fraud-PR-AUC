# Data Fusion Contest 2026 — Задача 1 "Страж"

Решение для задачи [Data Fusion Contest 2026 - Страж](https://ods.ai/competitions/data-fusion2026-guardian).

Это нейросетевое sequence-based anti-fraud решение, которое для каждой операции клиента строит stable- и telemetry-эмбеддинги события, агрегирует с помощью **mean-pooling** историю событий по нескольким окнам разного размера, опционально добавляет session/future/label-history сигналы, а затем обучает **multitask**-модель с тремя головами (`red`, `suspicious`, `red-vs-yellow`).

## Что делает решение на верхнем уровне

Пайплайн состоит из следующих этапов:

1. **Загрузка исходных данных**.
2. **Преобразование сырых колонок** в набор категориальных и числовых признаков.
3. Опционально: построение **каузальных label-history признаков** по прошлым `yellow/red` событиям клиента.
4. Построение **последовательностных сегментов** по клиентам.
5. Для каждого CV-фолда:
   - построение словарей категорий только по **видимой истории**,
   - обучение модели,
   - расчёт OOF-предсказаний на validation-месяце,
   - расчёт test-предсказаний fold-моделью.
6. Усреднение test-предсказаний по fold-моделям в **CV-ensemble**.
7. Опционально: обучение **final model** на всём train.
8. Сохранение артефактов:
   - логи,
   - OOF,
   - fold predictions,
   - test predictions,
   - checkpoint'ы,
   - итоговые CSV для сабмита,
   - JSON с метриками и конфигурацией запуска.

## Требования

### Python
Рекомендуется Python 3.10+.

### Основные зависимости

- `numpy`
- `polars`
- `torch`
- `scikit-learn`
- `python-dateutil`

Пример установки:

```bash
pip install numpy polars torch scikit-learn python-dateutil
```

## Входные данные

Скрипт ожидает один parquet-файл, содержащий **все периоды** (`pretrain/train/pretest/test`) в единой таблице.

Для его подготовки необходимо запустить скрипт:
```bash
python prepare_data.py <path_to_raw_data_dir> <path_to_output_file>
```

В результате будет создан один файл, в котором будут все исходные данные в более компактной форме.

### Важные предположения

1. Файл **отсортирован** по:
   - `customer_id`
   - `event_dttm`

2. В parquet присутствуют как минимум следующие столбцы:

- `customer_id` – id клиента банка
- `event_id` — id операции
- `event_dttm` – дата/время операции
- `event_type_nm` – тип операции
- `event_desc` – закодировнное описание операции
- `channel_indicator_type` – канал совершения операции
- `channel_indicator_subtype` – подтип канала совершения операции
- `operaton_amt` — сумма операции в рублях
- `currency_iso_cd` – валюта операции (0 - если значение неизвестно)
- `mcc_code` – группа MCC merchant_category_code (0 - если неизвестна)
- `pos_cd` – закодированное значение point of sale condition code (0 - если значение неизвестно)
- `accept_language` – язык заголовка http запроса
- `browser_language` – язык браузера (0 - если значение неизвестно)
- `timezone` — часовой пояс (0 - если значение неизвестно)
- `session_id` – идентификатор сессии
- `operating_system_type` – закодированное значение операционной системы (0 - если значение неизвестно)
- `battery` – заряд устройства (значение <0, если заряд батареи недоступен)
- `developer_tools` — настройки разработчика (флаг на устройстве) (0 - если значение неизвестно)
- `phone_voip_call_state` – флаг события VoIP-звонка во время проведения операции (0 - если значение неизвестно)
- `web_rdp_connection` – флаг наличия удаленного управления над устройством (0 - если значение неизвестно)
- `compromised` – наличие Root-доступа на устройстве (0 - если значение неизвестно)
- `device_system_version` – версия операционной системы в формате 10000 * X + 100 * Y + Z для версии X.Y.Z (0 - если значение неизвестно)
- `device_system_version_parts` - число частей в версии ОС (например, для 10.2 равняется 2)
- `screen_size_1` – разрешение экрана для стороны 1 (0 - если значение неизвестно)
- `screen_size_2` – разрешение экрана для стороны 2 (0 - если значение неизвестно)
- `original_index` - индекс записи в исходном датасете pretrain/train/pretest/test
- `dataset` - идентификатор датасета (0 - pretrain, 1 - train, 2 - pretest, 3 - test)
- `target` - значение таргета (0 - неизвестно, 1 - обычная операция, 2 - подозрительная подтвержденная операция, 3 - подозрительная неподтвержденная операция)

## Быстрый старт

Чтобы получить скор **0.131** - **0.134** на лидерборде, запустите `train_last_n_pooling.py` со следующими параметрами:

```bash
python train_last_n_pooling.py \
  --input-path <path_to_input_data> \
  --output-dir results/{date}/{time} \
  --cv-mode sliding \
  --threads 8 \
  --gpu \
  --target1-sample-frac 0.05 \
  --max-epochs 50 \
  --patience 6 \
  --lr 0.0003 \
  --batch-size-gpu 256 \
  --multitask-inference-blend 0 \
  --one-hot-max-cardinality 32 \
  --use-session-branch \
  --use-label-history \
  --use-future-branches
```

Данный скрипт обучает 5 моделей на разных фолдах, а затем создает финальное предсказание.

Время обучения одной модели - около 1 часа на RTX 5090. Соответственно общее время работы скрипта - около 5 часов.

Для работы желательно 64-128 Гб RAM и около 8 Гб VRAM.

Можно убрать опции `--use-session-branch` и `--use-future-branches`, это значительно ускорит обучение и снизит требования к RAM, финальный скор снизится при этом на 0.002 - 0.004.

Для тех у кого нет времени/ресурсов на запуск решения, в папке `results` приведены результаты запуска (скор на ЛБ - 0.1318048177). В папке есть сам submission, а также OOF-предсказания.

## Полная логика решения

### 1. Препроцессинг

После загрузки parquet из сырых колонок строятся признаки двух типов:

- **категориальные** (`BASE_CAT_COLS`)
- **числовые** (`BASE_NUM_COLS`)
- при `--use-label-history` к числовым добавляются ещё `LABEL_HISTORY_NUM_COLS`

#### 1.1. Категориальные признаки

Используются следующие категориальные признаки:

| Признак | Источник | Комментарий |
|---|---|---|
| `event_desc` | raw | код операции |
| `event_type_nm` | raw | тип операции |
| `channel_indicator_type` | raw | тип канала |
| `channel_indicator_sub_type` | raw | подтип канала |
| `mcc_code` | raw | MCC-группа |
| `pos_cd` | raw | POS code |
| `currency_iso_cd` | raw | валюта |
| `browser_language` | raw | язык браузера |
| `timezone` | raw | timezone |
| `operating_system_type` | raw | ОС |
| `developer_tools` | raw | dev tools flag |
| `phone_voip_call_state` | raw | VoIP flag |
| `web_rdp_connection` | raw | remote desktop flag |
| `compromised` | raw | root/jailbreak-like признак |
| `accept_lang_primary` | derived | primary язык из `accept_language`, захешированный |

##### Как строится `accept_lang_primary`

Из `accept_language`:
- строка приводится к lower-case,
- берётся первый токен до запятой,
- отбрасывается `;q=...`,
- отбрасывается региональный суффикс (`ru-RU -> ru`),
- затем строка хешируется.

Примеры:
- `ru-RU,ru;q=0.9,...` -> `ru`
- `en-US,en;q=0.9,...` -> `en`

#### 1.2. Числовые признаки

Используются следующие числовые признаки:

| Признак | Формула / смысл |
|---|---|
| `amount_log` | `log1p(clean_amount)` |
| `amount_missing` | `1`, если `operaton_amt` отсутствует |
| `battery_value` | заряд батареи в `[0,1]`, если доступен |
| `battery_available` | `1`, если батарея известна и `>= 0` |
| `battery_neg` | `-battery`, если батарея была отрицательной |
| `device_version_norm` | `device_system_version / 100000` |
| `device_parts_norm` | `device_system_version_parts / 4` |
| `screen1_norm` | `screen_size_1 / 4000` |
| `screen2_norm` | `screen_size_2 / 4000` |
| `session_present` | `1`, если session_id известен |
| `delta_time_log` | `log1p(time_since_prev_event_sec)` по клиенту |
| `hour_sin` / `hour_cos` | циклическое кодирование часа |
| `dow_sin` / `dow_cos` | циклическое кодирование дня недели |
| `is_weekend` | `1`, если выходной |
| `accept_lang_missing` | `1`, если `accept_language` пуст |
| `browser_language_known` | `1`, если `browser_language > 0` |
| `operating_system_type_known` | `1`, если ОС известна |
| `developer_tools_known` | `1`, если признак dev tools известен |
| `phone_voip_call_state_known` | `1`, если VoIP признак известен |
| `web_rdp_connection_known` | `1`, если remote desktop признак известен |
| `compromised_known` | `1`, если compromised известен |
| `device_version_known` | `1`, если версия устройства известна |
| `screen_known` | `1`, если хотя бы одно измерение экрана известно |
| `accept_lang_primary_known` | `1`, если удалось выделить primary language |
| `telemetry_available_any` | `1`, если хоть какая-то telemetry/device информация доступна |

##### Особенности расчётов

- `event_dttm` принудительно приводится к `ns` (`nanoseconds`) до вычисления временных разниц.
- `delta_time_log` считается **внутри клиента**.
- Отрицательные / null интервалы времени и аномальные значения клипуются.
- Сумма операции очищается:
  - `null` или отрицательные значения заменяются на `0` для `amount_log`.

#### 1.3. Опциональные label-history признаки

Если включён `--use-label-history`, к числовым признакам добавляются:

- `lh_prev_yellow_count_log`
- `lh_prev_red_count_log`
- `lh_since_last_yellow_log`
- `lh_since_last_red_log`
- `lh_prev_suspicious_count_log`
- `lh_since_last_suspicious_log`

##### Что они означают

Для каждой операции клиента вычисляются **каузальные** признаки по прошлым размеченным событиям этого же клиента:

- сколько `yellow` уже было до текущего момента;
- сколько `red` уже было до текущего момента;
- сколько `yellow|red` (suspicious) было до текущего момента;
- сколько времени прошло с последнего `yellow`;
- сколько времени прошло с последнего `red`;
- сколько времени прошло с последнего suspicious.

Количество логарифмируется через `log1p`.

##### Важно

Эти признаки используют только **прошлые события** клиента:
текущая строка сама в свою историю не попадает.

### 2. Кодирование категорий

Категориальные признаки кодируются **отдельно для каждого CV-фолда**:

- словарь категорий строится только по строкам с `event_dttm < valid_start`,
- все unseen значения маппятся в `UNK`.

#### Схема кодов

- `0` — padding
- `1` — unknown / unseen category
- `2...` — известные категории

#### Embedding vs one-hot

Для каждого категориального признака используется один из двух режимов:

- **one-hot**, если реальная мощность признака `<= --one-hot-max-cardinality`
- **embedding**, если мощность больше порога

Если `--one-hot-max-cardinality 0`, то **все категории** идут через embeddings.

### 3. Построение последовательностных сегментов

Вместо обработки всей истории клиента одной гигантской последовательностью скрипт разбивает данные на **сегменты**.

Это нужно в первую очередь для экономии памяти.

Для каждого клиента история режется на блоки вида:
- левый контекст: до `last_n` прошлых событий,
- предсказываемый кусок: до `max_pred_seq_len` событий,
- правый контекст: только если включены future branches.

Каждый сегмент описывается четырьмя индексами:

- `ctx_start`
- `pred_start`
- `pred_end`
- `ctx_end`

Таким образом, в сегмент входят:

1. **исторический контекст** до `pred_start`,
2. **предсказываемая часть** `[pred_start, pred_end)`,
3. опционально — **future context** после `pred_end`, если включены future branches.

Важный нюанс: соседние сегменты клиента могут перекрываться по контексту. Это нормально.

#### Управляющие параметры

- `--last-n`  
  максимум прошлых событий клиента, которые можно дать как контекст в сегменте.

- `--max-pred-seq-len`  
  максимум операций внутри одного предсказываемого блока.

- `--future-windows` и `--future-max-hours`  
  ограничивают, насколько далеко вправо сегмент может быть расширен для future branches.

#### Ограничение

`last_n` должен быть не меньше `max(history_windows)`.

### 4. Temporal CV

CV выполняется **по месяцам** внутри train-периода.

#### Период в коде
- train начинается с `2024-10-01`
- первая validation-точка — `2025-01-01`
- train заканчивается на `2025-06-01`

#### Поддерживаемые режимы

##### `--cv-mode expanding`

Обучение идёт от фиксированного старта:

- Fold 0:
  - train: `2024-10-01 ... 2025-01-01`
  - valid: `2025-01-01 ... 2025-02-01`
- Fold 1:
  - train: `2024-10-01 ... 2025-02-01`
  - valid: `2025-02-01 ... 2025-03-01`
- и т.д.

##### `--cv-mode sliding`

Используется скользящее окно длины `--sliding-train-months`.

### 5. Downsampling класса `target == 1` (зеленые события)

Параметр:

```bash
--target1-sample-frac
```

Позволяет случайно оставлять только часть строк с `target == 1`:

- в train-фите,
- в ES-subset,
- в final model.

#### Важно

`yellow` и `red` не сэмплируются — они всегда остаются.

## Архитектура модели

### Общая идея

Модель состоит из:

1. **двух event-encoder веток**:
   - stable
   - telemetry

2. **набора branch-level pooled context блоков**:
   - stable past
   - telemetry past
   - (optional) session
   - (optional) future stable
   - (optional) future telemetry

3. **Дополнительный прямой numeric block**
   - (optional) label history

3. **трёх классификационных голов**:
   - `red`
   - `suspicious`
   - `red vs yellow`

### 1. Event-level encoders

#### 1.1 Разделение признаков на stable и telemetry

`stable` — это признаки, которые ближе к самой операции и её бизнес-контексту:

- `event_desc`
- `event_type_nm`
- channel
- `mcc_code`
- `pos_cd`
- `currency_iso_cd`
- `timezone`
- amount/time features

Это то, что:
- обычно чаще заполнено,
- более устойчиво во времени,
- сильнее связано с "типом операции и поведением клиента".

`telemetry` — это device/app/security признаки:

- browser / OS
- screen / battery
- dev tools
- VoIP
- RDP
- compromised
- accept-language
- session presence

Это то, что:
- может быть часто неизвестно,
- может меняться из-за продукта/технологии,
- часто имеет другой тип дрейфа,
- нередко более шумное, но иногда очень сильное.

Зачем их разделять?

##### 1. У них разная природа

Операционный контекст и device/security контекст — это не одно и то же.

##### 2. У них разная доступность

Telemetry часто отсутствует (например, в pretrain).  
В коде это ещё специально учитывается: telemetry history считается только по строкам, где telemetry вообще доступна.

##### 3. У них разный drift

Stable-признаки обычно более устойчивы.
Telemetry-признаки могут сильно меняться из-за:
- обновлений приложения,
- новых устройств,
- изменения сбора данных.

##### 4. Им полезно дать разные encoder'ы

Один MLP может лучше учить transactional часть, другой — device/security часть.

#### 1.2. Stable branch

Использует "стабильные" признаки операции.

##### Категориальные stable-признаки

- `event_desc`
- `event_type_nm`
- `channel_indicator_type`
- `channel_indicator_sub_type`
- `mcc_code`
- `pos_cd`
- `currency_iso_cd`
- `timezone`

##### Числовые stable-признаки

- `amount_log`
- `amount_missing`
- `delta_time_log`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `is_weekend`

#### 1.3. Telemetry branch

Использует device/app/security признаки.

##### Категориальные telemetry-признаки

- `browser_language`
- `operating_system_type`
- `developer_tools`
- `phone_voip_call_state`
- `web_rdp_connection`
- `compromised`
- `accept_lang_primary`

##### Числовые telemetry-признаки

- `battery_value`
- `battery_available`
- `battery_neg`
- `device_version_norm`
- `device_parts_norm`
- `screen1_norm`
- `screen2_norm`
- `session_present`
- `accept_lang_missing`
- `browser_language_known`
- `operating_system_type_known`
- `developer_tools_known`
- `phone_voip_call_state_known`
- `web_rdp_connection_known`
- `compromised_known`
- `device_version_known`
- `screen_known`
- `accept_lang_primary_known`
- `telemetry_available_any`

#### Архитектура энкодеров

Обе ветки используют MLP вида:

```text
Linear(input_dim -> hidden_dim)
GELU
Dropout
Linear(hidden_dim -> event_dim)
GELU
```

На выходе:
- `stable_event_emb` размера `event_dim`
- `telemetry_event_emb` размера `event_dim`

### 2. Branches и окна

Модель агрегирует контекст **по нескольким окнам**.

#### History windows

Параметр:
```bash
--history-windows 3,10,30,100
```

Это означает, что для каждого события модель отдельно смотрит на:
- последние 3 события,
- последние 10,
- последние 30,
- последние 100.

#### Future windows

Параметр:
```bash
--future-windows 3,10,30,100
```

Аналогично, но для future branches:
- следующие 3,
- 10,
- 30,
- 100 событий,

при этом future-события дополнительно ограничиваются по времени:
```bash
--future-max-hours
```

### 3. Mean pooling

Mean-pooling строит "средний исторический контекст":
- каким обычно был клиент в этом масштабе окна,
- каким обычно был telemetry/device контекст в этом масштабе,
- какой тип операции в среднем был рядом.

### 4. Активные branch'и

#### 4.1. `stable_past`

История по stable-event embedding.

- Используются **все прошлые валидные события** в пределах окна.
- Для каждого окна строится средний embedding истории.

#### 4.2. `telemetry_past`

История по telemetry-event embedding.

- В историю берутся только события, где: `telemetry_available_any == 1`

То есть telemetry-контекст считается только по строкам, где telemetry вообще присутствует.

#### 4.3. `session` (опционально)

Включается через:

```bash
--use-session-branch
```

Это история по telemetry-event embedding, но только по событиям:
- из прошлого,
- в пределах history-window,
- с тем же `session_id`,
- `session_id != 0`.

Домашнее задание: реализовать stable_session_branch (для stable-событий).

#### 4.4. `future_stable` (опционально)

Включается через:

```bash
--use-future-branches
```

Смотрит на **будущие события** в рамках:
- future-window по числу событий,
- и `future-max-hours` по времени.

Использует stable-event embedding.

#### 4.5. `future_telemetry` (опционально)

То же, что future_stable, но:
- использует telemetry-event embedding,
- future-история ограничивается событиями с `telemetry_available_any == 1`.

#### 4.6. `label_history` (опционально)

Включается через:

```bash
--use-label-history
```

Это не pooled branch, а прямой блок числовых признаков: 6 label-history колонок просто добавляются в общий fusion-вектор.

### 5. Какие признаки branch строит на каждом окне

Для каждой window-агрегации branch строит 4 блока:

1. **`hist`**  
   усреднённый embedding истории

2. **`diff = reference - hist`**  
   отклонение текущего события от исторического контекста

3. **`prod = reference * hist`**  
   поэлементное произведение

4. **`frac`**  
   доля заполненности окна:
   ```text
   frac = min(1, hist_count / window)
   ```

#### Для branch'ей с `include_current=True`

В начало признаков branch также добавляется embedding текущего события (`reference_emb`):
- `stable_past`
- `telemetry_past`

#### Для branch'ей с `include_current=False`

Текущий embedding отдельно не добавляется:
- `session`
- `future_stable`
- `future_telemetry`

### 6. Как branch'и объединяются

Все активные branch-блоки конкатенируются:

```text
fused =
    [stable_past |
     telemetry_past |
     optional(session) |
     optional(future_stable) |
     optional(future_telemetry) |
     optional(label_history)]
```

После этого `fused` подаётся на три головы.

## Классификационные головы

### 1. Голова `red`

Основная задача: `target == 3` против всех остальных размеченных.

Выход:
- `red_logits`
- `p_red_main = sigmoid(red_logits)`

### 2. Голова `suspicious`

Промежуточная задача: `(target == 2 or target == 3)` против обычных.

Выход:
- `suspicious_logits`
- `p_suspicious = sigmoid(suspicious_logits)`

### 3. Голова `ry`

Подзадача внутри suspicious: `red vs yellow`

Выход:
- `ry_logits`
- `p_red_given_suspicious = sigmoid(ry_logits)`

Эта голова **не** обучается на green-строках.

### 4. Финальный скор на инференсе

Считается две оценки вероятности red:

#### Основная:

```text
p_red_main
```

#### Auxiliary:

```text
p_red_aux = p_suspicious * p_red_given_suspicious
```

#### Итог:

```text
final_score =
    (1 - multitask_inference_blend) * p_red_main
    + multitask_inference_blend * p_red_aux
```

Параметр:
```bash
--multitask-inference-blend
```

## Обучение

### Функции потерь

Используются три `BCEWithLogitsLoss`:

1. `red loss`
2. `suspicious loss`
3. `ry loss`

#### Где считаются

- `red loss` — по всем размеченным строкам (`target > 0`)
- `suspicious loss` — по всем размеченным строкам
- `ry loss` — только по строкам, где `target in {2,3}`

#### Веса positive class

Для каждой задачи считается свой `pos_weight` по обучающей выборке fold'а.

Кап: не больше `100`.

### Оптимизатор

Используется: `AdamW`

Параметры:
- `--lr`
- `--weight-decay`

### Early stopping

На каждом fold'е ранняя остановка идёт по `final_ap` на ES-subset.

#### ES-subset

Это sampled-подмножество validation-месяца:
- все `target != 1` остаются,
- `target == 1` могут быть downsampled через `--target1-sample-frac`.

#### Контролируемые параметры

- `--patience`
- `--max-epochs`

## Final model

Если включён:

```bash
--train-final-model
```

то после CV запускается ещё один этап:

1. словари категорий строятся по всем строкам с временем `< 2025-06-01`,
2. модель обучается на **всём train**,
3. число эпох берётся как:
   - `median(best_epochs)` по CV-fold'ам,
   - если fold'ов нет, то `max_epochs`.

Для final model:
- early stopping не используется,
- test-предсказания сохраняются отдельно,
- создаётся отдельный финальный CSV.

## Параметры CLI

Ниже перечислены все поддерживаемые параметры.

### Обязательные

#### `--input-path`

Путь к входному parquet-файлу.

Пример:
```bash
--input-path data/full_dataset.parquet
```

#### `--output-dir`

Папка для артефактов.

Поддерживаются шаблоны:
- `{date}`
- `{time}`

Пример:
```bash
--output-dir runs/{date}_{time}
```

#### `--cv-mode`

Режим CV:
- `expanding`
- `sliding`

### Флаги запуска

#### `--train-final-model`

После CV дополнительно обучить финальную модель на всём train.

#### `--gpu`

Использовать CUDA, если доступна.

### Общие параметры

#### `--random-state`

Seed для:
- numpy
- python random
- torch

Default: `42`

#### `--threads`

Количество CPU thread'ов для torch.

Default: `os.cpu_count()`

### Параметры окон и сегментов

#### `--history-windows`

Список past windows по числу событий.

Default: `3,10,30,100`

#### `--future-windows`

Список future windows по числу событий.

Default: `3,10,30,100`

#### `--last-n`

Максимум прошлых событий в сегменте.

Default: `128`

#### `--max-pred-seq-len`

Максимум предсказываемых позиций в одном сегменте.

Default: `256`

### Параметры обучения

#### `--max-epochs`

Максимум эпох.

Default: `8`

#### `--patience`

Patience для early stopping.

Default: `3`

#### `--lr`

Learning rate.

Default: `1e-3`

#### `--weight-decay`

Weight decay для AdamW.

Default: `1e-4`

#### `--dropout`

Dropout внутри MLP.

Default: `0.15`

#### `--hidden-dim`

Скрытая размерность MLP.

Default: `128`

#### `--event-dim`

Размерность event embedding.

Default: `32`

#### `--batch-size-cpu`

Batch size на CPU.

Default: `16`

#### `--batch-size-gpu`

Batch size на GPU.

Default: `64`

#### `--grad-clip`

Клиппинг градиента.

Default: `1.0`

#### `--train-ap-every-n-epochs`

Как часто считать AP на train.

Default: `3`

### Параметры CV

#### `--sliding-train-months`
Длина окна train для `cv-mode=sliding`.

Default: `3`

### Параметры multitask

#### `--aux-loss-weight-suspicious`

Вес suspicious loss.

Default: `0.5`

#### `--aux-loss-weight-red-yellow`

Вес `red vs yellow` loss.

Default: `0.5`

#### `--multitask-inference-blend`

Вес auxiliary red-score на инференсе.

Default: `0.4`

### Session branch

#### `--use-session-branch`

Включить same-session branch.

#### `--session-branch-weight`

Вес pooled-фичей session branch.

Default: `0.5`

### Future branches

#### `--use-future-branches`

Включить future branches.

#### `--future-max-hours`

Максимальный future horizon в часах.

Default: `24.0`

#### `--future-branch-weight`

Вес pooled-фичей future branch.

Default: `0.5`

### Кодирование категорий

#### `--one-hot-max-cardinality`

Если мощность категориального признака `<=` этого порога, он кодируется one-hot, иначе embedding.

Default: `0`

То есть по умолчанию one-hot отключён.

### Downsampling

#### `--target1-sample-frac`

Доля строк `target == 1`, которая остаётся в обучении.

Default: `1.0`

### Label history

#### `--use-label-history`
Добавить 6 каузальных признаков по прошлым размеченным suspicious-событиям клиента.

## Артефакты после запуска

Скрипт создаёт папку `output-dir` и сохраняет в неё следующие файлы.

### Основные

#### `debug.log`

Подробный лог выполнения.

#### `run_options.json`

Фактические CLI-параметры запуска.

#### `results.json`

Итоговый summary:
- агрегированные метрики,
- конфигурация,
- ссылки на артефакты,
- компактный summary по fold'ам.

#### `fold_metrics_detailed.json`

Подробные метрики по каждому fold'у:
- train/valid размеры,
- AP-метрики,
- параметры категориального кодирования,
- `epoch_history`,
- пути к моделям и т.д.

### Предсказания

#### `oof_predictions.parquet`

OOF-предсказания по train-части, покрытой validation-fold'ами.

Поля:
- `event_id`
- `customer_id`
- `event_dttm`
- `target`
- `predict`
- `predict_final`
- `predict_red_main`
- `predict_suspicious`
- `predict_red_given_suspicious`
- `fold_idx`
- `oof_available`

#### `valid_predictions_fold_XX.parquet`

Validation-предсказания конкретного fold'а.

#### `test_predictions_fold_XX.parquet`

Test-предсказания конкретной fold-модели.

#### `test_predictions_cv_ensemble.parquet`

Усреднённый ансамбль test-предсказаний по всем fold'ам.

#### `submission_cv_ensemble.csv`

CSV для отправки на лидерборд:
- `event_id`
- `predict`

### Модели

#### `models/fold_XX.pt`

Checkpoint fold-модели.

Внутри:
- `state_dict`
- описание категорий
- cardinailities
- vocab values
- model config
- training summary

#### `models/final_model.pt`

Checkpoint финальной модели, если включён `--train-final-model`.

### Артефакты final model

#### `test_predictions_final_model.parquet`

Предсказания финальной модели на test.

#### `submission_final_model.csv`

Итоговый CSV сабмит финальной модели.

#### `final_model_training.json`

Подробная информация о финальном обучении:
- число эпох,
- training summary,
- category metadata,
- пути к модели.

## Практические замечания

Рекомендую всегда задавать `--multitask-inference-blend` равным 0. Так как `yellow` события выделяются гораздо хуже `red` событий, поэтому при бленде общий скор **падает**.

При запуске с `--target1-sample-frac` равным 0.05 модель обучается лишь на малой доле `green` событий. Домашнее задание: подумайте что с этим можно сделать.

## Краткая схема модели

```text
raw parquet
   ↓
polars preprocessing
   ↓
base categorical + base numerical features
   ↓
optional label-history features
   ↓
sequence segmentation by customer
   ↓
per-fold category vocab / encoding
   ↓
stable event encoder ─┐
                      ├─ past pooling over history windows
telemetry encoder ────┘
                      ├─ optional session pooling
                      ├─ optional future pooling
                      └─ optional label-history block
   ↓
concatenate branch features
   ↓
3 heads:
  - red
  - suspicious
  - red_vs_yellow
   ↓
final blended score
```

## Что читать в коде в первую очередь

Логично смотреть код в таком порядке:

1. `parse_args()`  
   какие есть режимы и флаги

2. `load_and_preprocess()`  
   какие признаки строятся

3. `build_segments()`  
   как формируются последовательности

4. `build_encoded_store()`  
   как кодируются категории по fold'ам

5. `MultiTaskTwoBranchMultiWindowModel`  
   вся архитектура модели

6. `run_training_job()`  
   как устроены обучение, early stopping и метрики

7. `main()`  
   как связаны CV, OOF, test ensemble и final model

## Как можно далее улучшить решение

### 1. Улучшить входные признаки

Сейчас решение в основном опирается на:
- текущую операцию,
- sequence context,
- label history.

Можно добавить более явные handcrafted-признаки:

#### Поведенческие отклонения клиента

Например:
- это новый `event_desc` для клиента,
- это новый `mcc`,
- это новый channel,
- это новый timezone,
- это новый device family,
- amount deviation относительно типичного значения клиента,
- как часто клиент делает такой `event_desc`,
- как часто такой `mcc`,
- как давно видел такой channel,
- насколько нетипична сумма,
- и т.д.

#### Burst-признаки

Например:
- число событий за 5 минут / 1 час / 24 часа,
- число suspicious-похожих событий рядом,
- скорость серии операций.

### 2. Развить pooling

Сейчас реализован только mean-pooling.

Можно попробовать:
- time-decay pooling,
- attention pooling,
- gated pooling,
- learnable mixture of windows.

### 3. Развить архитектуру

Возможные варианты:
- разные размерности для stable и telemetry,
- отдельная stable-session ветка,
- общий interaction block между stable и telemetry,
- реализовать branch gating (которые будут оключать ветки в зависимости от текущего события).

А может вообще использовать блоки трансформеров?

### 4. Новые типы лоссов

Возможно стоит попробовать другие типы лоссов: pairwise, focal losses?
