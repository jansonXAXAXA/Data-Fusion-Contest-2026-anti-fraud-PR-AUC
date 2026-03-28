# Data Fusion 2026 - задача "Страж", паблик решение команды ICEQ

<p align="center"><img src=images/logo.png width=450px></p>

## Краткое описание задачи

Соревнование по классификации банковских операций. Цель - предсказать, какие операции **не были подтверждены клиентами** (RED - фрод).

- **Данные**: 200M+ операций для 100K клиентов за 1.5 года
- **Метрика**: PR-AUC (`sklearn.metrics.average_precision_score`)
- **Классы**: RED  (51K, фрод), YELLOW  (36K, подтверждённые подозрительные), GREEN  (остальное)
- **Лидерборд**: Public = недели 1,3,5 (30%), Private = остальные 7 недель (70%)
- **Лучший скор**: **0.1414 LB** (public)

![EDA Overview](images/eda_overview.png)

---

## Конфигурация

```
CPU: Intel Core Ultra 285k (но запустится решение может и на другом процессоре, основная нагрузка идет на гпу)
GPU: NVIDIA RTX 5060 Ti 16GB
RAM: 64GB
```

---

## Структура нашего решения

```
current_pipeline/
 run_catboost.py          # Основной CatBoost пайплайн (фичи + обучение + блендинг)
 run_coles.py             # Обучение CoLES -> извлечение юзер-эмбеддингов
 run_coles_refit.py       # Refit CatBoost (+ CoLES эмбеддинги) -> создание сабмита
 coles_seed_fb50.csv      # Лучший сабмит (0.1414)
```

### Трехэтапный пайплайн

1. **`run_catboost.py`** - генерация фичей (3 партиции), обучение 4 CatBoost моделей, подбор весов блендинга, сохранение кэша:
   - `cache/features_part_{1,2,3}.parquet` - фичи
   - `cache/features_all.parquet` + `cache/test_features.parquet` - трейн/тест pandas
   - `cache/v5_config.json` - лучшие итерации, веса блендинга, alpha FB

2. **`run_coles.py`** - обучения модели Co-LES -> сохранение эмбеддингов клиентов в `cache/coles_embeddings.parquet`, сохранение последовательностей клиентов в `cache/customer_sequences.parquet`, сохранение модели в `cache/coles_model.pt`

3. **`run_coles_refit.py`** - загрузка кэша, присоединение CoLES эмбеддингов, рефит 4 моделей * 3 сида, генерация сабмитов с разными параметрами alpha (обычно, 50 дает лучший результат)

---

## Стек моделей

### 4 суб-модели CatBoost

| Модель | Таргет | Данные | Назначение |
|--------|--------|--------|------------|
| **MAIN** | target_bin (RED=1) | Весь сэмплированный трейн | Основной предсказатель фрода |
| **SUSPICIOUS** | is_labeled (RED∪YELLOW=1) | Весь сэмплированный трейн | «Подозрительна ли операция?» |
| **RED\|SUSP** | target_bin | Только размеченные (RED∪YELLOW) | «Если подозрительная - RED?» |
| **FEEDBACK** | target_bin | Все (с FB-фичами) | Использует историю меток клиента |

### Формула бленда:

```
product = logit(sigmoid(SUSP) * sigmoid(RED|SUSP))

CB_blend = w_main * MAIN + w_rec * RECENT + w_prod * product
           (типично: 0.35 / 0.0 / 0.65)

FB_inject = (1 - α) * rank(CB_blend) + α * rank(FB_model)
            (только для клиентов с историей меток, α ≈ 0.5)
```

### CoLES эмбеддинги

![CoLES Training](images/coles_training.png)

- **Архитектура**: 12 категориальных эмбеддингов (8-dim) + 1 числовой (amt_log) -> GRU(256-dim, 2 слоя) -> mean-pool -> linear -> 256-dim вектор клиента
- **Лосс**: NT-Xent, две случайные 64-событийные подпоследовательности одного клиента = положительная пара
- **Обучение**: 15 эпох, batch=256, AdamW + cosine schedule, на 100K клиентах, 177M событий
- **Результат**: 256-dim «поведенческий отпечаток» клиента -> фича в CatBoost

### 3-seed усреднение

Рефит на полном трейне с сидами 42, 123, 777 -> усреднение предсказаний для стабильности.

---

## Группы фичей

![Feature Categories](images/feature_categories.png)

### 1. Категориальные (15)

`customer_id`, `event_type_nm`, `event_desc`, `channel_indicator_type`, `channel_indicator_sub_type`, `currency_iso_cd`, `mcc_code_i`, `pos_cd`, `timezone`, `operating_system_type`, `phone_voip_call_state`, `web_rdp_connection`, `developer_tools_i`, `compromised_i`, `prev_mcc_code_i`

### 2. Sums (8)

`amt`, `amt_log_abs`, `amt_abs`, `amt_is_negative`, `amt_delta_prev`, `amt_to_prev_mean`, `amt_zscore`, `amt_profile_zscore`

### 3. Time-aware (11)

`hour`, `weekday`, `day`, `month`, `is_weekend`, `is_night`, `is_night_early`, `hour_sin`, `hour_cos`, `event_day_number`, `day_of_year`

### 4. Device-aware (10)

`battery_pct`, `os_ver_major`, `screen_w`, `screen_h`, `screen_pixels`, `screen_ratio`, `voip_rdp_combo`, `any_risk_flag`, `compromised_devtools`, `lang_mismatch`

### 5. Последовательные / customer-based (20+)

`cust_prev_events`, `cust_prev_amt_mean`, `cust_prev_amt_std`, `sec_since_prev_event`, `cnt_prev_same_type/desc/mcc/subtype/session`, `sec_since_prev_same_type/desc/mcc`, `events_before_today`, `mcc_changed`, `session_changed`, `cust_events_per_day`

### 6. Rolling speed-based (12)

`cnt_15min`, `cnt_1h`, `cnt_6h`, `cnt_24h`, `cnt_7d`, `amt_sum_15min`, `amt_sum_1h`, `amt_sum_24h`, `amt_ratio_24h`, `burst_ratio_1h_24h`, `spend_concentration_1h`, `burst_ratio_15m_1h`

### 7. Current (transaction-day features) (3)

`today_total_amount`, `today_max_amount`, `today_amt_vs_daily_norm`

### 8. Rolling mean (скользящие средние) по кол-ву транзакций (5)

`amt_avg_5`, `amt_avg_10`, `amt_avg_50`, `amt_avg_100`, `amt_momentum` (= avg_5 − avg_50)

### 9. Markov MCC (2)

`markov_mcc_prob`, `markov_mcc_surprise` - аномалия перехода между категориями MCC

### 10. Log-count (14)

`{col}_log_cnt` для 14 категориальных колонок - частота поведения клиент * категория

### 11. Смена устройства (3)

`os_changed`, `screen_changed`, `tz_changed`

### 12. Паттерны пропусков (null values) (11)

`null_{col}` для 10 устройственных колонок + `null_device_count`

### 13. Профили клиентов из претрейна (9)

`profile_txn_count`, `profile_amt_mean/std/median/max/p95`, `profile_n_unique_mcc`, `profile_hour_mean`, `profile_avg_daily_txns`

+ производные: `amt_over_profile_mean`, `amt_over_profile_p95`, `amt_profile_zscore`

### 14. Prior fraud ratios (22)

`prior_{key}_cnt/red_rate/red_share` для 7 колонок + 4 интеракционные пары

### 15. Фидбек-фичи (история пользователей - только для тех, у которых есть лейблы в трейне) (14, только FB-модель)

`cust_prev_red/yellow_lbl_cnt`, `rates`, `flags`, `sec_since_prev_red/yellow_lbl`, `per-desc counts`

### 16. Фичи детекции типов фрода (15)

**Социальная инженерия**:
- `voip_cnt_15min`, `had_voip_before_txn` - VoIP-активность за 15 минут до транзакции (клиент разговаривал с кем-то по телефону за 15 минут до транзакции / в момент транзакции - риск мошенничества)
- `is_round_amount` - круглые суммы (кратные 500/1000)
- `sec_since_session_start` - время от начала сессии

**Изменение поведения пользователя (friendly fraud)**:
- `hour_mean_5`, `hour_drift` - среднее значение часа за 5 транзакций vs профильное
- `amt_drift_5` - среднее за 5 транзакций vs историческое среднее
- `mcc_diversity_ratio` - уникальные MCC за 24ч / профильное кол-во
- `ch_dev_combo_log_cnt` - редкость комбинации канал+устройство

**Физическая кража карты**:
- `unique_mcc_1h`, `unique_mcc_24h` - уникальные MCC за окна
- `mcc_scatter_ratio` - unique_mcc_1h / unique_mcc_24h
- `tz_jump_magnitude` - абсолютный скачок часового пояса
- `impossible_travel` - скачок TZ > 2 за < 1 час
- `tz_change_cnt_24h` - кол-во смен часового пояса за 24ч
- `velocity_ratio_1h` - cnt_1h / профильное среднее дневных транзакций

### 17. CoLES эмбеддинги (256)

`coles_0` ... `coles_255` - CoLES-эмбеддинги клиентов

---

## Ключевые пункты решения

- **Сэмплирование негативов**: недавние зелёные 1:5, старые 1:15
- **Веса классов**: RED=10, YELLOW=2.5, GREEN=1.0
- **Валидация**: трейн < 2025-04-15, вал = случайный день (Random-Last-Day validation) для каждого клиента в промежутке[2025-04-15, 2025-06-01)
- **REFIT**: refit на полном трейне (без early stopping), используя best_iter из валидации
- **Блендинг**: RawFormulaVal (логиты), не вероятности; rank-average (среднее по ранку)
- **FB-инъекция (инъекция предсказаний фидбек-модели)**: условная - только для клиентов с историей размеченных событий

---

## Результаты экспериментов

![Score Progression](images/score_progression.png)

![Feature Importance](images/feature_importance.png)

## Что работает 

| # | Метод | Эффект |
|---|-------|--------|
| 1 | **FB (Feedback) инъекция** для клиентов с историей меток | +0.015 LB (самый большой буст) |
| 2 | **`customer_id` как cat_feature** в CatBoost | +0.02 LB |
| 3 | **CoLES самообучение** (контрастивный GRU -> 256-dim отпечаток клиента) | +0.003 LB |
| 4 | **3-seed усреднение** (42, 123, 777) | Снижает дисперсию рефита |
| 5 | **Product decomposition** P(RED) = P(susp) * P(RED\|susp) | Разделение задачи |
| 6 | **Априоры** на данных до валидации | Без утечки |
| 7 | **Rank-based блендинг** (не raw scores) | Нормализация масштабов |
| 8 | **REFIT на полном трейне** с best_iter | Больше данных -> лучше |
| 9 | **Null-паттерны** | Определяют тип платформы |
| 10 | **Марковская MCC-аномалия** | Удивление перехода между MCC |
| 11 | **15-мин агрегаты** | Ловят card-testing атаки |
| 12 | **Внутридневные фичи** | Другой угол vs скользящие окна |
| 13 | **Скользящие средние** (amt_avg_5/10/50/100) | Контекст последних транзакций |
| 14 | **Фичи детекции типов фрода** (VoIP, MCC scatter, impossible travel) | 0.1384 -> 0.1414 |

---

![Experiments](images/experiments.png)

## Что НЕ работает 

| # | Метод | Проблема |
|---|-------|----------|
| 1 | **GRU для предсказания фрода** | 0.079 LB - сигнал слишком разбавлен (0.04% размеченных) |
| 2 | **Interaction target encoding** | Прямая утечка таргета - высокий CV, обвал LB |
| 3 | **Pseudo-labeling** (слабой моделью 0.13) | 87% неверных меток -> confirmation bias |
| 4 | **Isolation Forest** | anomaly ≠ fraud, слишком много FP |
| 5 | **Temporal features** (event_day_number, day_of_year, month) в CatBoost | Переобучение на временные периоды |
| 6 | **LGB в финальном бленде** | Ухудшает LB: 0.138 -> 0.135 |
| 7 | **Customer-level smoothing** (pred = 0.7*pred + 0.3*cust_mean) | Стирает per-event FB сигнал |
| 8 | **Geometric blends** | Rank-average работает, geometric нет |
| 9 | **Focal Loss на CatBoost GPU** | Не поддерживается - только CPU, слишком медленно |
| 10 | **Pseudo-labeling из blend_0152** (top-500 -> pseudo-RED) | 0.1379 vs 0.1384 (−0.0005) |
| 11 | **Downsampling** (уменьшение негативов) | Чем больше данных - тем лучше |
| 12 | **Training window truncation** (1нед, 2нед, 1мес) | Полный трейн всегда побеждает |
| 13 | **Device graph / time-decay features** | Минимальный эффект (+0.001–0.003 val, 0 на LB) |
| 14 | **Self-labeling / co-training** (2x CatBoost) | Модели недостаточно уверены |
| 15 | **FB alpha > 0.5** | Переобучение на валидации |
| 16 | **Temporal Target Encoding** (expanding mean) | Не дало прироста, скор упал |

---

## Error Analysis: что модель пропускает

![Error Analysis](images/error_analysis.png)

Пропущенные фроды - это «тихие» мошенники: маленькие суммы (173K vs 5.5M пойманных), нет burst-паттерна, нет VoIP, обычные MCC. Они выглядят как нормальные транзакции. Для их поимки нужен анализ **отклонений от поведенческой нормы клиента**.

---

## Быстрый старт

### Вариант 1: Готовый сабмит (0 минут)
Скачайте `coles_seed_fb50.csv` и загрузите на лидерборд → **0.1414 PR-AUC**

### Вариант 2: С готовыми весами (~24 мин GPU)
```bash
# Фичи уже в кэше, CoLES эмбеддинги готовы
# Только refit CatBoost (3 seeds x 4 модели)
python run_coles_refit.py
```

### Вариант 3: Полный пайплайн (~60 мин)
```bash
python run_catboost.py       # 40 мин - фичи + обучение
python run_coles.py          # 11 мин - CoLES pre-training
python run_coles_refit.py    # 24 мин - refit + submission
```

### Предобученные веса
- `coles_model.pt` - обученный CoLES энкодер (2.9 MB, в репо)
- [`coles_embeddings.parquet`](https://disk.yandex.ru/d/u1A9SP00Tyn9aQ) - 256-dim эмбеддинги для 100K клиентов (181 MB, Яндекс.Диск)
- `coles_seed_fb50.csv` - готовый сабмит (22 MB, в репо)

---

## Скоры

| Модель / Бленд | Public LB | Примечание |
|---|---|---|
| CB + CoLES + FB + фичи типов фрода + 3-seed | **0.1414** |  Текущий лучший |
| CB ensemble + FB (соло) | 0.1384 | CatBoost + feedback-инъекция |
| CB ensemble | 0.118 | Базовый CatBoost |

---

## Подробный Writeup

Полная история решения с детальным описанием всех экспериментов: [WRITEUP.md](WRITEUP.md)

---

*Built with CatBoost, PyTorch, Polars by team ICEQ*
