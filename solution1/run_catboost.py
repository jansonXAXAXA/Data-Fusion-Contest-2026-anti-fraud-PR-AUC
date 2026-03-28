"""CatBoost пайплайн (только трейн, валидация и сохранение в кеш.
Нужно рефитнуть на полных данных (и еще + CoLES эмбеддинги) в run_coles_refit.py"""

import gc, os, sys, time, math, warnings, functools
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

_SOL1 = Path(__file__).resolve().parent
if str(_SOL1) not in sys.path:
    sys.path.insert(0, str(_SOL1))

# ── Конфигурация ──
from paths import CACHE_DIR, DATA_DIR, SUBMISSIONS

SEED = 42
USE_GPU = True
FORCE_REBUILD = True

NEG_SAMPLE_MOD_RECENT = 5
NEG_SAMPLE_MOD_OLD = 15
NEG_SAMPLE_BORDER = "2025-04-01 00:00:00"

VAL_WINDOW_START = pd.Timestamp("2025-04-15")
VAL_WINDOW_END = pd.Timestamp("2025-06-01")
RECENT_BORDER = pd.Timestamp("2025-02-01")

log = functools.partial(print, flush=True)

# ── Списки колонок ──
BASE_COLS = [
    "customer_id", "event_id", "event_dttm", "event_type_nm", "event_desc",
    "channel_indicator_type", "channel_indicator_sub_type", "operaton_amt", "currency_iso_cd",
    "mcc_code", "pos_cd", "timezone", "session_id", "operating_system_type",
    "battery", "device_system_version", "screen_size", "developer_tools",
    "phone_voip_call_state", "web_rdp_connection", "compromised",
    "accept_language", "browser_language",
]

CAT_COLS = [
    "customer_id", "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd",
    "timezone", "operating_system_type", "phone_voip_call_state", "web_rdp_connection",
    "developer_tools_i", "compromised_i", "prev_mcc_code_i",
]

# Колонки для log_cnt цикла (частотные поведенческие фичи)
LOG_CNT_COLS = [
    "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd", "timezone",
    "session_id", "operating_system_type", "developer_tools_i", "compromised_i",
    "phone_voip_call_state", "web_rdp_connection",
]

# Фичи истории обратной связи (используются ТОЛЬКО FB-моделью, не основными)
FB_FEATURE_COLS = [
    "cust_prev_red_lbl_cnt", "cust_prev_yellow_lbl_cnt", "cust_prev_labeled_cnt",
    "cust_prev_red_lbl_rate", "cust_prev_yellow_lbl_rate", "cust_prev_susp_lbl_rate",
    "cust_prev_any_red_flag", "cust_prev_any_yellow_flag",
    "sec_since_prev_red_lbl", "sec_since_prev_yellow_lbl",
    "cnt_prev_labeled_same_desc", "cnt_prev_red_same_desc_lbl",
    "cnt_prev_yellow_same_desc_lbl", "red_rate_prev_same_desc_lbl",
]

META_COLS = ["event_id", "period", "event_ts", "is_train_sample", "is_test", "train_target_raw", "target_bin"]

labels_lf = pl.scan_parquet(DATA_DIR / "train_labels.parquet")
labels_df = pl.read_parquet(DATA_DIR / "train_labels.parquet")
log(f"Labels: {labels_df.shape}")


# Генерация фичей
def _period_frames(part_id):
    custs_lf = pl.scan_parquet(DATA_DIR / f"pretrain_part_{part_id}.parquet").select("customer_id").unique()
    pretrain = pl.scan_parquet(DATA_DIR / f"pretrain_part_{part_id}.parquet").select(BASE_COLS).with_columns(pl.lit("pretrain").alias("period"))
    train = pl.scan_parquet(DATA_DIR / f"train_part_{part_id}.parquet").select(BASE_COLS).with_columns(pl.lit("train").alias("period"))
    pretest = pl.scan_parquet(DATA_DIR / "pretest.parquet").select(BASE_COLS).join(custs_lf, on="customer_id", how="inner").with_columns(pl.lit("pretest").alias("period"))
    test = pl.scan_parquet(DATA_DIR / "test.parquet").select(BASE_COLS).join(custs_lf, on="customer_id", how="inner").with_columns(pl.lit("test").alias("period"))
    return pl.concat([pretrain, train, pretest, test], how="vertical_relaxed")


def _build_profiles(part_id):
    lf = pl.scan_parquet(DATA_DIR / f"pretrain_part_{part_id}.parquet")
    lf = lf.with_columns([
        pl.col("operaton_amt").cast(pl.Float64).alias("amt"),
        pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_i"),
        pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("_ts"),
    ])
    return lf.group_by("customer_id").agg([
        pl.len().cast(pl.Int32).alias("profile_txn_count"),
        pl.col("amt").mean().cast(pl.Float32).alias("profile_amt_mean"),
        pl.col("amt").std().cast(pl.Float32).alias("profile_amt_std"),
        pl.col("amt").median().cast(pl.Float32).alias("profile_amt_median"),
        pl.col("amt").max().cast(pl.Float32).alias("profile_amt_max"),
        pl.col("amt").quantile(0.95).cast(pl.Float32).alias("profile_amt_p95"),
        # Доп. профильные фичи для upgrade plan
        pl.col("mcc_i").n_unique().cast(pl.Int16).alias("profile_n_unique_mcc"),
        pl.col("_ts").dt.hour().mean().cast(pl.Float32).alias("profile_hour_mean"),
    ]).with_columns([
        pl.col("profile_amt_std").fill_null(0.0),
        pl.col("profile_hour_mean").fill_null(12.0),
        # Среднее кол-во транзакций в день: total / ~365 дней претрейна
        (pl.col("profile_txn_count").cast(pl.Float32) / 365.0).alias("profile_avg_daily_txns"),
    ]).collect()


def build_features_part(part_id, force=False):
    out_path = CACHE_DIR / f"features_part_{part_id}.parquet"
    if out_path.exists() and not force:
        log(f"[part {part_id}] cache hit")
        return out_path

    log(f"[part {part_id}] building features...")
    lf = _period_frames(part_id)
    TWO_PI = 2.0 * math.pi

    # Парсинг колонок
    lf = lf.with_columns([
        pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("event_ts"),
        pl.col("operaton_amt").cast(pl.Float64).alias("amt"),
        pl.col("session_id").cast(pl.Int64, strict=False).fill_null(-1).alias("session_id"),
        pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1),
        pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1),
        pl.col("channel_indicator_type").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("channel_indicator_sub_type").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("currency_iso_cd").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("timezone").cast(pl.Int32, strict=False).fill_null(-1),
        pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1),
        pl.col("phone_voip_call_state").cast(pl.Int8, strict=False).fill_null(-1),
        pl.col("web_rdp_connection").cast(pl.Int8, strict=False).fill_null(-1),
        pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),
        pl.col("battery").str.extract(r"(\d{1,3})", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("battery_pct"),
        pl.col("device_system_version").str.extract(r"^(\d+)", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("os_ver_major"),
        pl.col("screen_size").str.extract(r"^(\d+)", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("screen_w"),
        pl.col("screen_size").str.extract(r"x(\d+)$", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("screen_h"),
        pl.col("developer_tools").cast(pl.Int8, strict=False).fill_null(-1).alias("developer_tools_i"),
        pl.col("compromised").cast(pl.Int8, strict=False).fill_null(-1).alias("compromised_i"),
        pl.when(pl.col("accept_language").is_not_null() & pl.col("browser_language").is_not_null())
          .then((pl.col("accept_language") != pl.col("browser_language")).cast(pl.Int8))
          .otherwise(pl.lit(-1).cast(pl.Int8)).alias("lang_mismatch"),
    ]).drop(["event_dttm", "operaton_amt", "mcc_code", "battery", "device_system_version",
             "screen_size", "developer_tools", "compromised", "accept_language", "browser_language"])
    lf = lf.sort(["customer_id", "event_ts", "event_id"])

    # Метки + сэмплирование
    lf = lf.join(labels_lf, on="event_id", how="left")
    lf = lf.with_columns(
        pl.when(pl.col("period") == "train")
          .then(pl.when(pl.col("target").is_null()).then(pl.lit(-1)).otherwise(pl.col("target")))
          .otherwise(pl.lit(None)).alias("train_target_raw")
    )

    border_expr = pl.lit(NEG_SAMPLE_BORDER).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    lf = lf.with_columns(
        ((pl.col("period") == "train") & (pl.col("train_target_raw") == -1) &
         (((pl.col("event_ts") >= border_expr) & ((pl.struct(["event_id", "customer_id"]).hash(seed=SEED) % NEG_SAMPLE_MOD_RECENT) == 0)) |
          ((pl.col("event_ts") < border_expr) & ((pl.struct(["event_id", "customer_id"]).hash(seed=SEED + 17) % NEG_SAMPLE_MOD_OLD) == 0))))
        .alias("keep_green")
    )

    # Временные фичи
    lf = lf.with_columns([
        ((pl.col("period") == "train") & ((pl.col("train_target_raw") != -1) | pl.col("keep_green"))).alias("is_train_sample"),
        (pl.col("period") == "test").alias("is_test"),
        pl.col("event_ts").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("event_ts").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("event_ts").dt.day().cast(pl.Int8).alias("day"),
        pl.col("event_ts").dt.month().cast(pl.Int8).alias("month"),
        (pl.col("event_ts").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        ((pl.col("event_ts").dt.hour() >= 22) | (pl.col("event_ts").dt.hour() < 6)).cast(pl.Int8).alias("is_night"),
        (pl.col("event_ts").dt.hour() < 6).cast(pl.Int8).alias("is_night_early"),
        (pl.col("event_ts").dt.hour().cast(pl.Float32) * (TWO_PI / 24.0)).sin().cast(pl.Float32).alias("hour_sin"),
        (pl.col("event_ts").dt.hour().cast(pl.Float32) * (TWO_PI / 24.0)).cos().cast(pl.Float32).alias("hour_cos"),
        (pl.col("event_ts").dt.epoch("s") // 86400).cast(pl.Int32).alias("event_day_number"),
        pl.col("event_ts").dt.ordinal_day().cast(pl.Int16).alias("day_of_year"),
        pl.col("event_ts").dt.date().alias("event_date"),
        pl.col("amt").abs().log1p().cast(pl.Float32).alias("amt_log_abs"),
        pl.col("amt").abs().cast(pl.Float32).alias("amt_abs"),
        (pl.col("amt") < 0).cast(pl.Int8).alias("amt_is_negative"),
        (pl.col("screen_w").cast(pl.Int32) * pl.col("screen_h").cast(pl.Int32)).alias("screen_pixels"),
        pl.when((pl.col("screen_h") > 0) & (pl.col("screen_w") > 0))
          .then(pl.col("screen_w").cast(pl.Float32) / pl.col("screen_h").cast(pl.Float32))
          .otherwise(0.0).alias("screen_ratio"),
    ])

    # Комбинации устройств
    lf = lf.with_columns([
        ((pl.col("phone_voip_call_state") == 1) & (pl.col("web_rdp_connection") == 1)).cast(pl.Int8).alias("voip_rdp_combo"),
        ((pl.col("phone_voip_call_state") == 1) | (pl.col("web_rdp_connection") == 1) |
         (pl.col("compromised_i") == 1) | (pl.col("developer_tools_i") == 1)).cast(pl.Int8).alias("any_risk_flag"),
        ((pl.col("compromised_i") == 1) & (pl.col("developer_tools_i") == 1)).cast(pl.Int8).alias("compromised_devtools"),
    ])

    # Смена устройства
    lf = lf.with_columns([
        pl.col("operating_system_type").shift(1).over("customer_id").alias("_prev_os"),
        pl.col("screen_w").shift(1).over("customer_id").alias("_prev_sw"),
        pl.col("timezone").shift(1).over("customer_id").alias("_prev_tz"),
    ])
    # Сохраняем _prev_tz_raw для вычисления tz_jump_magnitude ниже
    lf = lf.with_columns([
        ((pl.col("operating_system_type") != pl.col("_prev_os")) & pl.col("_prev_os").is_not_null()).cast(pl.Int8).alias("os_changed"),
        ((pl.col("screen_w") != pl.col("_prev_sw")) & pl.col("_prev_sw").is_not_null()).cast(pl.Int8).alias("screen_changed"),
        ((pl.col("timezone") != pl.col("_prev_tz")) & pl.col("_prev_tz").is_not_null()).cast(pl.Int8).alias("tz_changed"),
        pl.col("_prev_tz").alias("_prev_tz_raw"),
    ]).drop(["_prev_os", "_prev_sw", "_prev_tz"])

    # ── B2: редкость комбинации канал+устройство ──
    lf = lf.with_columns(
        (pl.col("channel_indicator_sub_type").cast(pl.Utf8) + "_" + pl.col("operating_system_type").cast(pl.Utf8)).alias("_ch_dev_combo")
    )
    lf = lf.with_columns(
        (pl.cum_count("event_id").over(["customer_id", "_ch_dev_combo"]) - 1).clip(lower_bound=0).log1p().cast(pl.Float32).alias("ch_dev_combo_log_cnt")
    )
    lf = lf.drop("_ch_dev_combo")

    # --- Флаги истории обратной связи (из меток, только train период) ---
    lf = lf.with_columns([
        ((pl.col("period") == "train") & (pl.col("train_target_raw") == 1)).cast(pl.Int8).alias("is_red_lbl"),
        ((pl.col("period") == "train") & (pl.col("train_target_raw") == 0)).cast(pl.Int8).alias("is_yellow_lbl"),
    ])
    lf = lf.with_columns(
        (pl.col("is_red_lbl") + pl.col("is_yellow_lbl")).cast(pl.Int8).alias("is_labeled_fb")
    )

    # Последовательные фичи
    lf = lf.with_columns([
        pl.cum_count("event_id").over("customer_id").cast(pl.Int32).alias("cust_event_idx"),
        pl.col("amt").cum_sum().over("customer_id").alias("cust_cum_amt"),
        (pl.col("amt") * pl.col("amt")).cum_sum().over("customer_id").alias("cust_cum_amt_sq"),
        pl.col("event_ts").shift(1).over("customer_id").alias("prev_event_ts"),
        pl.col("amt").shift(1).over("customer_id").alias("prev_amt"),
        pl.col("mcc_code_i").shift(1).over("customer_id").fill_null(-1).alias("prev_mcc_code_i"),
        pl.col("session_id").shift(1).over("customer_id").fill_null(-1).alias("prev_session_id"),
        (pl.cum_count("event_id").over(["customer_id", "event_type_nm"]) - 1).cast(pl.Int16).alias("cnt_prev_same_type"),
        (pl.cum_count("event_id").over(["customer_id", "event_desc"]) - 1).cast(pl.Int16).alias("cnt_prev_same_desc"),
        (pl.cum_count("event_id").over(["customer_id", "mcc_code_i"]) - 1).cast(pl.Int16).alias("cnt_prev_same_mcc"),
        (pl.cum_count("event_id").over(["customer_id", "channel_indicator_sub_type"]) - 1).cast(pl.Int16).alias("cnt_prev_same_subtype"),
        (pl.cum_count("event_id").over(["customer_id", "session_id"]) - 1).cast(pl.Int16).alias("cnt_prev_same_session"),
        pl.col("event_ts").shift(1).over(["customer_id", "event_type_nm"]).alias("prev_same_type_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "event_desc"]).alias("prev_same_desc_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "mcc_code_i"]).alias("prev_same_mcc_ts"),
        pl.col("event_ts").first().over("customer_id").alias("cust_first_ts"),
        # Кумулятивные счётчики истории обратной связи
        pl.col("is_red_lbl").cum_sum().over("customer_id").cast(pl.Int32).alias("cust_red_lbl_cum"),
        pl.col("is_yellow_lbl").cum_sum().over("customer_id").cast(pl.Int32).alias("cust_yellow_lbl_cum"),
        pl.col("is_labeled_fb").cum_sum().over("customer_id").cast(pl.Int32).alias("cust_labeled_fb_cum"),
        pl.col("is_red_lbl").cum_sum().over(["customer_id", "event_desc"]).cast(pl.Int16).alias("desc_red_lbl_cum"),
        pl.col("is_yellow_lbl").cum_sum().over(["customer_id", "event_desc"]).cast(pl.Int16).alias("desc_yellow_lbl_cum"),
        pl.col("is_labeled_fb").cum_sum().over(["customer_id", "event_desc"]).cast(pl.Int16).alias("desc_labeled_fb_cum"),
        # Таймстампы размеченных событий для расчёта time-since
        pl.when(pl.col("is_red_lbl") == 1).then(pl.col("event_ts")).otherwise(None).alias("red_lbl_ts"),
        pl.when(pl.col("is_yellow_lbl") == 1).then(pl.col("event_ts")).otherwise(None).alias("yellow_lbl_ts"),
    ])

    # Forward-fill таймстампов последних red/yellow меток (shift для исключения текущего)
    lf = lf.with_columns([
        pl.col("red_lbl_ts").shift(1).over("customer_id").alias("prev_red_lbl_ts"),
        pl.col("yellow_lbl_ts").shift(1).over("customer_id").alias("prev_yellow_lbl_ts"),
    ])
    lf = lf.with_columns([
        pl.col("prev_red_lbl_ts").forward_fill().over("customer_id").alias("prev_red_lbl_ts"),
        pl.col("prev_yellow_lbl_ts").forward_fill().over("customer_id").alias("prev_yellow_lbl_ts"),
    ])

    # Производные фичи
    lf = lf.with_columns([
        (pl.col("cust_event_idx") - 1).cast(pl.Int32).alias("cust_prev_events"),
        pl.when(pl.col("cust_event_idx") > 1)
          .then((pl.col("cust_cum_amt") - pl.col("amt")) / (pl.col("cust_event_idx") - 1))
          .otherwise(0.0).cast(pl.Float32).alias("cust_prev_amt_mean"),
        pl.when(pl.col("prev_event_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_event_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_event"),
        (pl.col("amt") - pl.col("prev_amt").fill_null(0.0)).cast(pl.Float32).alias("amt_delta_prev"),
        pl.when(pl.col("prev_same_type_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_type_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_type"),
        pl.when(pl.col("prev_same_desc_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_desc_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_desc"),
        pl.when(pl.col("prev_same_mcc_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_mcc_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_same_mcc"),
        (pl.cum_count("event_id").over(["customer_id", "event_date"]) - 1).cast(pl.Int16).alias("events_before_today"),
        (pl.col("mcc_code_i") != pl.col("prev_mcc_code_i")).cast(pl.Int8).alias("mcc_changed"),
        (pl.col("session_id") != pl.col("prev_session_id")).cast(pl.Int8).alias("session_changed"),
        pl.when((pl.col("event_ts") - pl.col("cust_first_ts")).dt.total_days() > 0)
          .then(pl.col("cust_event_idx").cast(pl.Float32) / (pl.col("event_ts") - pl.col("cust_first_ts")).dt.total_days().cast(pl.Float32))
          .otherwise(1.0).cast(pl.Float32).alias("cust_events_per_day"),
        # ── Внутридневные фичи ── обнуляются в полночь
        (pl.col("amt_abs").cum_sum().over(["customer_id", "event_date"]) - pl.col("amt_abs")).cast(pl.Float32).alias("today_total_amount"),
        # ── A3: круглые суммы (признак соц. инженерии) ──
        ((pl.col("amt_abs") % 1000 == 0) | (pl.col("amt_abs") % 500 == 0)).cast(pl.Int8).alias("is_round_amount"),
        # ── A2: поведение в сессии ──
        pl.when(pl.col("event_ts").first().over(["customer_id", "session_id"]).is_not_null())
          .then((pl.col("event_ts") - pl.col("event_ts").first().over(["customer_id", "session_id"])).dt.total_seconds())
          .otherwise(0).cast(pl.Int32).alias("sec_since_session_start"),
        # ── C3: величина скачка часового пояса ──
        (pl.col("timezone").cast(pl.Int32) - pl.col("_prev_tz_raw").cast(pl.Int32)).abs().fill_null(0).cast(pl.Int16).alias("tz_jump_magnitude"),
    ])

    # Внутридневной: today_max_amount (shift + cum_max для исключения текущего события)
    lf = lf.with_columns(
        pl.col("amt_abs").shift(1).over(["customer_id", "event_date"]).fill_null(0.0).alias("_prev_amt_today")
    )
    lf = lf.with_columns(
        pl.col("_prev_amt_today").cum_max().over(["customer_id", "event_date"]).cast(pl.Float32).alias("today_max_amount"),
    )
    lf = lf.drop(["_prev_amt_today", "_prev_tz_raw"])

    # Отношение суммы текущей транзакции к дневному обороту
    lf = lf.with_columns(
        pl.when(pl.col("today_total_amount") > 1.0)
          .then(pl.col("amt_abs") / pl.col("today_total_amount"))
          .otherwise(1.0).cast(pl.Float32).alias("today_amt_vs_daily_norm")
    )

    # ── C3: невозможное перемещение (производная фича) ──
    lf = lf.with_columns(
        ((pl.col("tz_jump_magnitude") > 2) & (pl.col("sec_since_prev_event") > 0) & (pl.col("sec_since_prev_event") < 3600)).cast(pl.Int8).alias("impossible_travel")
    )

    # История обратной связи: исключаем текущее событие (cum - current)
    lf = lf.with_columns([
        (pl.col("cust_red_lbl_cum") - pl.col("is_red_lbl")).cast(pl.Int32).alias("cust_prev_red_lbl_cnt"),
        (pl.col("cust_yellow_lbl_cum") - pl.col("is_yellow_lbl")).cast(pl.Int32).alias("cust_prev_yellow_lbl_cnt"),
        (pl.col("cust_labeled_fb_cum") - pl.col("is_labeled_fb")).cast(pl.Int32).alias("cust_prev_labeled_cnt"),
        (pl.col("desc_labeled_fb_cum") - pl.col("is_labeled_fb")).cast(pl.Int16).alias("cnt_prev_labeled_same_desc"),
        (pl.col("desc_red_lbl_cum") - pl.col("is_red_lbl")).cast(pl.Int16).alias("cnt_prev_red_same_desc_lbl"),
        (pl.col("desc_yellow_lbl_cum") - pl.col("is_yellow_lbl")).cast(pl.Int16).alias("cnt_prev_yellow_same_desc_lbl"),
        # Время с последнего размеченного события
        pl.when(pl.col("prev_red_lbl_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_red_lbl_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_red_lbl"),
        pl.when(pl.col("prev_yellow_lbl_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_yellow_lbl_ts")).dt.total_seconds())
          .otherwise(-1).cast(pl.Int32).alias("sec_since_prev_yellow_lbl"),
    ])

    # История обратной связи: доли и флаги
    lf = lf.with_columns([
        ((pl.col("cust_prev_red_lbl_cnt") + 0.1) / (pl.col("cust_prev_labeled_cnt") + 1.0)).cast(pl.Float32).alias("cust_prev_red_lbl_rate"),
        ((pl.col("cust_prev_yellow_lbl_cnt") + 0.1) / (pl.col("cust_prev_labeled_cnt") + 1.0)).cast(pl.Float32).alias("cust_prev_yellow_lbl_rate"),
        (((pl.col("cust_prev_red_lbl_cnt") + pl.col("cust_prev_yellow_lbl_cnt")) + 0.1) / (pl.col("cust_prev_events") + 1.0)).cast(pl.Float32).alias("cust_prev_susp_lbl_rate"),
        (pl.col("cust_prev_red_lbl_cnt") > 0).cast(pl.Int8).alias("cust_prev_any_red_flag"),
        (pl.col("cust_prev_yellow_lbl_cnt") > 0).cast(pl.Int8).alias("cust_prev_any_yellow_flag"),
        ((pl.col("cnt_prev_red_same_desc_lbl") + 0.1) / (pl.col("cnt_prev_labeled_same_desc") + 1.0)).cast(pl.Float32).alias("red_rate_prev_same_desc_lbl"),
    ])

    # Марковская вероятность MCC-перехода
    lf = lf.with_columns((pl.col("prev_mcc_code_i").cast(pl.Utf8) + "_" + pl.col("mcc_code_i").cast(pl.Utf8)).alias("_mcc_bigram"))
    lf = lf.with_columns([
        (pl.cum_count("event_id").over(["customer_id", "prev_mcc_code_i"]) - 1).cast(pl.Float32).alias("_prev_mcc_cnt"),
        (pl.cum_count("event_id").over(["customer_id", "_mcc_bigram"]) - 1).cast(pl.Float32).alias("_bigram_cnt"),
    ])
    lf = lf.with_columns((pl.col("_bigram_cnt") / pl.col("_prev_mcc_cnt").clip(1, None)).fill_null(0).cast(pl.Float32).alias("markov_mcc_prob"))
    lf = lf.with_columns((-pl.col("markov_mcc_prob").clip(1e-6, None).log()).cast(pl.Float32).alias("markov_mcc_surprise"))
    lf = lf.drop(["_mcc_bigram", "_prev_mcc_cnt", "_bigram_cnt"])

    # Log-count циклы: частота поведения клиент × категория (без текущего события)
    for col in LOG_CNT_COLS:
        lf = lf.with_columns(
            (pl.cum_count("event_id").over(["customer_id", col]) - 1).clip(lower_bound=0).log1p().cast(pl.Float32).alias(f"{col}_log_cnt")
        )

    # Бегущее стд. откл. + аномалия
    lf = lf.with_columns(
        pl.when(pl.col("cust_event_idx") > 2)
          .then(((pl.col("cust_cum_amt_sq") - pl.col("amt") * pl.col("amt")) / (pl.col("cust_event_idx") - 1) - pl.col("cust_prev_amt_mean") * pl.col("cust_prev_amt_mean")).clip(lower_bound=0).sqrt())
          .otherwise(0.0).cast(pl.Float32).alias("cust_prev_amt_std")
    )
    lf = lf.with_columns([
        pl.when(pl.col("cust_prev_amt_mean").abs() > 0.01).then(pl.col("amt") / pl.col("cust_prev_amt_mean")).otherwise(0.0).cast(pl.Float32).alias("amt_to_prev_mean"),
        pl.when(pl.col("cust_prev_amt_std") > 1.0).then((pl.col("amt") - pl.col("cust_prev_amt_mean")) / pl.col("cust_prev_amt_std")).otherwise(0.0).cast(pl.Float32).alias("amt_zscore"),
    ])

    # ── Скользящие средние по кол-ву транзакций ──
    # Сдвиг на 1 для исключения текущего события (строго каузально)
    log(f"  [part {part_id}] скользящие средние...")
    for n in [5, 10, 50, 100]:
        lf = lf.with_columns(
            pl.col("amt").shift(1).rolling_mean(n).over("customer_id")
              .cast(pl.Float32).alias(f"amt_avg_{n}")
        )
    lf = lf.with_columns(
        (pl.col("amt_avg_5").fill_null(0.0) - pl.col("amt_avg_50").fill_null(0.0))
          .cast(pl.Float32).alias("amt_momentum")
    )

    # ── B1: поведенческий дрейф (короткое окно vs история) ──
    log(f"  [part {part_id}] фичи поведенческого дрейфа...")
    # Среднее значение часа за последние 5 событий (со сдвигом)
    lf = lf.with_columns([
        pl.col("hour").cast(pl.Float32).shift(1).rolling_mean(5).over("customer_id").cast(pl.Float32).alias("hour_mean_5"),
    ])


    # Целевая переменная
    lf = lf.with_columns(
        pl.when(pl.col("is_train_sample")).then((pl.col("train_target_raw") == 1).cast(pl.Int8)).otherwise(pl.lit(None)).alias("target_bin")
    )

    # Скользящая скорость (без pretrain)
    log(f"  [part {part_id}] скользящая скорость...")
    roll_df = lf.filter(pl.col("period") != "pretrain").select(
        ["event_id", "customer_id", "event_ts", "amt_abs", "phone_voip_call_state", "mcc_code_i", "tz_changed"]
    ).collect()
    roll_df = roll_df.sort(["customer_id", "event_ts"])

    # Окна подсчёта (15мин, 1ч, 6ч, 24ч, 7д)
    for window, suffix in [("15m", "15min"), ("1h", "1h"), ("6h", "6h"), ("1d", "24h"), ("7d", "7d")]:
        r = roll_df.rolling(index_column="event_ts", period=window, by="customer_id", closed="left").agg(pl.len().cast(pl.Int16).alias(f"cnt_{suffix}"))
        roll_df = roll_df.with_columns(r[f"cnt_{suffix}"])

    # Суммы сумм за окна (15мин, 1ч, 24ч)
    for window, suffix in [("15m", "15min"), ("1h", "1h"), ("1d", "24h")]:
        r = roll_df.rolling(index_column="event_ts", period=window, by="customer_id", closed="left").agg(pl.col("amt_abs").sum().cast(pl.Float32).alias(f"amt_sum_{suffix}"))
        roll_df = roll_df.with_columns(r[f"amt_sum_{suffix}"])

    # ── A1: VoIP за последние 15 минут ──
    r = roll_df.rolling(index_column="event_ts", period="15m", by="customer_id", closed="left").agg(
        pl.col("phone_voip_call_state").filter(pl.col("phone_voip_call_state") == 1).len().cast(pl.Int16).alias("voip_cnt_15min")
    )
    roll_df = roll_df.with_columns(r["voip_cnt_15min"])

    # ── C2: разброс MCC (уникальные MCC за 1ч и 24ч) ──
    for window, suffix in [("1h", "1h"), ("1d", "24h")]:
        r = roll_df.rolling(index_column="event_ts", period=window, by="customer_id", closed="left").agg(
            pl.col("mcc_code_i").n_unique().cast(pl.Int16).alias(f"unique_mcc_{suffix}")
        )
        roll_df = roll_df.with_columns(r[f"unique_mcc_{suffix}"])

    # ── B3: скорость смены часового пояса (кол-во за 24ч) ──
    r = roll_df.rolling(index_column="event_ts", period="1d", by="customer_id", closed="left").agg(
        pl.col("tz_changed").sum().cast(pl.Int16).alias("tz_change_cnt_24h")
    )
    roll_df = roll_df.with_columns(r["tz_change_cnt_24h"])

    roll_cols = [
        "event_id", "cnt_15min", "cnt_1h", "cnt_6h", "cnt_24h", "cnt_7d",
        "amt_sum_15min", "amt_sum_1h", "amt_sum_24h",
        "voip_cnt_15min", "unique_mcc_1h", "unique_mcc_24h", "tz_change_cnt_24h",
    ]
    lf = lf.join(roll_df.select(roll_cols).lazy(), on="event_id", how="left")
    del roll_df; gc.collect()

    lf = lf.with_columns([
        pl.when(pl.col("amt_sum_24h") > 1.0).then(pl.col("amt_abs") / pl.col("amt_sum_24h")).otherwise(1.0).cast(pl.Float32).alias("amt_ratio_24h"),
        pl.when(pl.col("cnt_24h") > 0).then(pl.col("cnt_1h").cast(pl.Float32) / pl.col("cnt_24h").cast(pl.Float32)).otherwise(0.0).cast(pl.Float32).alias("burst_ratio_1h_24h"),
        pl.when(pl.col("amt_sum_24h") > 1.0).then(pl.col("amt_sum_1h").fill_null(0.0) / pl.col("amt_sum_24h")).otherwise(0.0).cast(pl.Float32).alias("spend_concentration_1h"),
        pl.when(pl.col("cnt_1h") > 0).then(pl.col("cnt_15min").cast(pl.Float32) / pl.col("cnt_1h").cast(pl.Float32)).otherwise(0.0).cast(pl.Float32).alias("burst_ratio_15m_1h"),
        # Производные фичи
        (pl.col("voip_cnt_15min") > 0).cast(pl.Int8).alias("had_voip_before_txn"),
        pl.when(pl.col("unique_mcc_24h") > 0).then(pl.col("unique_mcc_1h").cast(pl.Float32) / pl.col("unique_mcc_24h").cast(pl.Float32)).otherwise(0.0).cast(pl.Float32).alias("mcc_scatter_ratio"),
    ])

    # Сбор и фильтрация
    FEATURE_COLS = [
        "customer_id", "event_type_nm", "event_desc", "channel_indicator_type",
        "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd", "timezone",
        "operating_system_type", "phone_voip_call_state", "web_rdp_connection",
        "developer_tools_i", "compromised_i",
        "amt", "amt_log_abs", "amt_abs", "amt_is_negative",
        "hour", "weekday", "day", "month", "is_weekend", "is_night", "is_night_early",
        "hour_sin", "hour_cos", "event_day_number", "day_of_year",
        "battery_pct", "os_ver_major", "screen_w", "screen_h", "screen_pixels", "screen_ratio",
        "session_id",
        "voip_rdp_combo", "any_risk_flag", "compromised_devtools", "lang_mismatch",
        "cust_prev_events", "cust_prev_amt_mean", "cust_prev_amt_std",
        "amt_to_prev_mean", "amt_zscore",
        "sec_since_prev_event", "amt_delta_prev",
        "cnt_prev_same_type", "cnt_prev_same_desc", "cnt_prev_same_mcc",
        "cnt_prev_same_subtype", "cnt_prev_same_session",
        "sec_since_prev_same_type", "sec_since_prev_same_desc", "sec_since_prev_same_mcc",
        "events_before_today",
        "mcc_changed", "prev_mcc_code_i", "session_changed",
        "cust_events_per_day",
        # Скользящая скорость
        "cnt_15min", "cnt_1h", "cnt_6h", "cnt_24h", "cnt_7d",
        "amt_sum_15min", "amt_sum_1h", "amt_sum_24h",
        "amt_ratio_24h", "burst_ratio_1h_24h", "spend_concentration_1h",
        "burst_ratio_15m_1h",
        # Внутридневные
        "today_total_amount", "today_max_amount", "today_amt_vs_daily_norm",
        # Скользящие средние
        "amt_avg_5", "amt_avg_10", "amt_avg_50", "amt_avg_100", "amt_momentum",
        # Смена устройства
        "os_changed", "screen_changed", "tz_changed",
        "markov_mcc_prob", "markov_mcc_surprise",
        # ── Фичи детекции типов фрода ──
        "is_round_amount",                                      # A3: соц. инженерия
        "sec_since_session_start",                               # A2: поведение в сессии
        "ch_dev_combo_log_cnt",                                  # B2: редкость канал+устройство
        "tz_jump_magnitude", "impossible_travel",                # C3: невозможное перемещение
        "hour_mean_5",                                           # B1: дрейф часа
        "voip_cnt_15min", "had_voip_before_txn",                # A1: VoIP до транзакции
        "unique_mcc_1h", "unique_mcc_24h", "mcc_scatter_ratio", # C2: разброс MCC
        "tz_change_cnt_24h",                                     # B3: скорость смены TZ
    ] + [f"{c}_log_cnt" for c in LOG_CNT_COLS]

    out_df = lf.filter(pl.col("is_train_sample") | pl.col("is_test")).select(META_COLS + FEATURE_COLS + FB_FEATURE_COLS).collect()

    # Профили клиентов из претрейна
    profiles = _build_profiles(part_id)
    out_df = out_df.join(profiles, on="customer_id", how="left")
    for c in profiles.columns:
        if c != "customer_id" and c in out_df.columns:
            out_df = out_df.with_columns(pl.col(c).fill_null(0.0))
    out_df = out_df.with_columns([
        pl.when(pl.col("profile_amt_mean").abs() > 0.01).then(pl.col("amt") / pl.col("profile_amt_mean")).otherwise(0.0).cast(pl.Float32).alias("amt_over_profile_mean"),
        pl.when(pl.col("profile_amt_p95").abs() > 0.01).then(pl.col("amt") / pl.col("profile_amt_p95")).otherwise(0.0).cast(pl.Float32).alias("amt_over_profile_p95"),
        pl.when(pl.col("profile_amt_std") > 1.0).then((pl.col("amt") - pl.col("profile_amt_mean")) / pl.col("profile_amt_std")).otherwise(0.0).cast(pl.Float32).alias("amt_profile_zscore"),
        # ── B1: дрейф суммы (среднее за 5 vs история) ──
        (pl.col("amt_avg_5").fill_null(0.0) - pl.col("cust_prev_amt_mean").fill_null(0.0)).cast(pl.Float32).alias("amt_drift_5"),
        # ── B1: дрейф часа (среднее за 5 vs профильное) ──
        (pl.col("hour_mean_5").fill_null(12.0) - pl.col("profile_hour_mean").fill_null(12.0)).abs().cast(pl.Float32).alias("hour_drift"),
        # ── B1: отношение разнообразия MCC ──
        pl.when(pl.col("profile_n_unique_mcc") > 0)
          .then(pl.col("unique_mcc_24h").fill_null(0).cast(pl.Float32) / pl.col("profile_n_unique_mcc").cast(pl.Float32))
          .otherwise(0.0).cast(pl.Float32).alias("mcc_diversity_ratio"),
        # ── C1: аномалия скорости трат ──
        pl.when(pl.col("profile_avg_daily_txns") > 0.1)
          .then(pl.col("cnt_1h").fill_null(0).cast(pl.Float32) / pl.col("profile_avg_daily_txns"))
          .otherwise(0.0).cast(pl.Float32).alias("velocity_ratio_1h"),
    ])
    del profiles; gc.collect()

    out_df.write_parquet(out_path, compression="zstd")
    log(f"[part {part_id}] done: {out_df.height:,} rows")
    del out_df; gc.collect()
    return out_path


# ══════════════════════════════════════════════════════════
# ОБУЧЕНИЕ
# ══════════════════════════════════════════════════════════
def make_weights(raw_target):
    return np.where(raw_target == 1, 10.0, np.where(raw_target == 0, 2.5, 1.0)).astype(np.float32)


def fit_cb(X_tr, y_tr, w_tr, X_val, y_val, w_val, cat_cols, params, use_gpu=True):
    p = params.copy()
    p.setdefault("eval_metric", "AUC")
    p.update({"loss_function": "Logloss", "random_seed": SEED, "allow_writing_files": False, "verbose": 200})
    if use_gpu:
        p.update({"task_type": "GPU", "devices": "0"})
    pool_tr = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_cols)
    pool_val = Pool(X_val, y_val, weight=w_val, cat_features=cat_cols)
    model = CatBoostClassifier(**p)
    model.fit(pool_tr, eval_set=pool_val, use_best_model=True)
    raw = model.predict(pool_val, prediction_type="RawFormulaVal")
    ap = average_precision_score(y_val, raw)
    bi = model.get_best_iteration() or p.get("iterations", 1000)
    log(f"  best_iter={bi}, val_PR-AUC={ap:.6f}")
    return model, bi, ap, p


def refit_cb(X, y, w, cat_cols, base_params, best_iter):
    p = base_params.copy()
    p.pop("od_type", None); p.pop("od_wait", None)
    p["iterations"] = int(max(300, best_iter))
    y_arr = np.asarray(y).ravel()
    w_arr = np.asarray(w, dtype=np.float32)
    if w_arr.ndim == 0:
        w_arr = np.full(len(y_arr), float(w_arr), dtype=np.float32)
    elif w_arr.shape[0] != len(y_arr):
        w_arr = np.full(len(y_arr), 1.0, dtype=np.float32)
    pool = Pool(X, y_arr, weight=w_arr, cat_features=cat_cols)
    model = CatBoostClassifier(**p)
    model.fit(pool, verbose=200)
    return model


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))

def _logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


# ══════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    # Генерация фичей
    log("ГЕНЕРАЦИЯ ФИЧЕЙ")
    paths = [build_features_part(i, force=FORCE_REBUILD) for i in [1, 2, 3]]
    features = pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")
    features = features.unique(subset=["event_id"])
    gc.collect()
    log(f"Features: {features.shape}")

    # Априорные вероятности — строго на данных ДО валидации, чтобы избежать утечки
    log("АПРИОРНЫЕ ВЕРОЯТНОСТИ")
    PRIOR_COLS = {
        "event_desc": pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1).alias("event_desc"),
        "mcc_code_i": pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),
        "timezone": pl.col("timezone").cast(pl.Int32, strict=False).fill_null(-1).alias("timezone"),
        "operating_system_type": pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1).alias("operating_system_type"),
        "channel_indicator_sub_type": pl.col("channel_indicator_sub_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_sub_type"),
        "event_type_nm": pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1).alias("event_type_nm"),
        "pos_cd": pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("pos_cd"),
    }
    prior_feat_cols = []
    val_border_str = str(VAL_WINDOW_START)
    for key, expr in PRIOR_COLS.items():
        lf_scan = pl.concat([
            pl.scan_parquet(DATA_DIR / f"train_part_{i}.parquet")
              .select([pl.col("event_id"), pl.col("event_dttm"), expr])
            for i in [1,2,3]
        ], how="vertical_relaxed")
        lf_scan = lf_scan.filter(
            pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False) < pl.lit(val_border_str).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        ).drop("event_dttm")
        total = lf_scan.group_by(key).len().rename({"len": f"prior_{key}_cnt"})
        labeled = lf_scan.join(labels_lf, on="event_id", how="inner").group_by(key).agg([pl.len().alias("_lbl"), pl.sum("target").cast(pl.Float64).alias("_red")])
        prior = total.join(labeled, on=key, how="left").with_columns([pl.col("_lbl").fill_null(0.0), pl.col("_red").fill_null(0.0)]).with_columns([
            ((pl.col("_red") + 1.0) / (pl.col(f"prior_{key}_cnt") + 200.0)).cast(pl.Float32).alias(f"prior_{key}_red_rate"),
            ((pl.col("_red") + 1.0) / (pl.col("_lbl") + 2.0)).cast(pl.Float32).alias(f"prior_{key}_red_share"),
        ]).select([key, f"prior_{key}_cnt", f"prior_{key}_red_rate", f"prior_{key}_red_share"]).collect()
        features = features.join(prior, on=key, how="left")
        prior_feat_cols.extend([c for c in prior.columns if c != key])

    # ── Интеракционные априоры ──
    log("ИНТЕРАКЦИОННЫЕ АПРИОРЫ")
    INTERACTIONS = [("mcc_code_i", "event_type_nm"), ("event_desc", "channel_indicator_sub_type"),
                    ("pos_cd", "event_type_nm"), ("mcc_code_i", "channel_indicator_sub_type")]
    for col_a, col_b in INTERACTIONS:
        ix_name = f"{col_a}x{col_b}"
        select_a = PRIOR_COLS.get(col_a, pl.col(col_a))
        select_b = PRIOR_COLS.get(col_b, pl.col(col_b))
        ix_lf = pl.concat([
            pl.scan_parquet(DATA_DIR / f"train_part_{i}.parquet")
              .select([pl.col("event_id"), pl.col("event_dttm"), select_a, select_b])
            for i in [1,2,3]
        ], how="vertical_relaxed")
        ix_lf = ix_lf.filter(
            pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            < pl.lit(val_border_str).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        ).drop("event_dttm")
        ix_total = ix_lf.group_by([col_a, col_b]).len().rename({"len": "_cnt"})
        ix_labeled = ix_lf.join(labels_lf, on="event_id", how="inner").group_by([col_a, col_b]).agg([
            pl.len().alias("_lbl"), pl.sum("target").cast(pl.Float64).alias("_red")
        ])
        ix_prior = ix_total.join(ix_labeled, on=[col_a, col_b], how="left").with_columns([
            pl.col("_lbl").fill_null(0.0), pl.col("_red").fill_null(0.0)
        ]).with_columns([
            ((pl.col("_red") + 1.0) / (pl.col("_cnt") + 200.0)).cast(pl.Float32).alias(f"prior_{ix_name}_red_rate"),
            ((pl.col("_red") + 1.0) / (pl.col("_lbl") + 2.0)).cast(pl.Float32).alias(f"prior_{ix_name}_red_share"),
        ]).select([col_a, col_b, f"prior_{ix_name}_red_rate", f"prior_{ix_name}_red_share"]).collect()
        features = features.join(ix_prior, on=[col_a, col_b], how="left")
        prior_feat_cols.extend([f"prior_{ix_name}_red_rate", f"prior_{ix_name}_red_share"])
        log(f"  {ix_name}: {ix_prior.height} combos")

    if prior_feat_cols:
        features = features.with_columns([pl.col(c).fill_null(pl.col(c).mean()).alias(c) for c in prior_feat_cols])

    # ── Фичи паттернов пропусков ──
    log("ФИЧИ ПАТТЕРНОВ ПРОПУСКОВ")
    null_device_cols = ["phone_voip_call_state", "web_rdp_connection", "timezone",
                        "operating_system_type", "developer_tools_i", "compromised_i",
                        "battery_pct", "os_ver_major", "screen_w", "screen_h"]
    null_exprs = [(pl.col(c) == -1).cast(pl.Int8).alias(f"null_{c}") for c in null_device_cols if c in features.columns]
    features = features.with_columns(null_exprs)
    null_count_cols = [f"null_{c}" for c in null_device_cols if c in features.columns]
    features = features.with_columns(
        sum(pl.col(c) for c in null_count_cols).cast(pl.Int8).alias("null_device_count")
    )
    log(f"  Added {len(null_exprs)+1} null features")

    # Конвертация в pandas
    log("ПОДГОТОВКА ДАННЫХ")
    train_pl = features.filter(pl.col("is_train_sample")).with_columns(pl.col("target_bin").cast(pl.Int8))
    test_pl = features.filter(pl.col("is_test"))
    log(f"Train: {train_pl.shape}, Test: {test_pl.shape}")

    train_df = train_pl.to_pandas()
    test_df = test_pl.to_pandas()
    del features, train_pl, test_pl; gc.collect()

    train_df["event_ts"] = pd.to_datetime(train_df["event_ts"])
    test_df["event_ts"] = pd.to_datetime(test_df["event_ts"])

    feature_cols = [c for c in train_df.columns if c not in META_COLS and c not in ["target", "keep_green", "event_date"] and c not in FB_FEATURE_COLS]
    fb_cols = [c for c in FB_FEATURE_COLS if c in train_df.columns]
    fb_feature_cols = feature_cols + fb_cols  # расширенный набор только для FB-модели
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for c in cat_cols:
        train_df[c] = train_df[c].fillna(-1).astype(np.int64)
        test_df[c] = test_df[c].fillna(-1).astype(np.int64)
    medians = train_df[num_cols].median(numeric_only=True)
    train_df[num_cols] = train_df[num_cols].fillna(medians)
    test_df[num_cols] = test_df[num_cols].fillna(medians)
    # Заполняем NaN FB-фичей отдельно (0.0 для неразмеченных клиентов)
    for c in fb_cols:
        train_df[c] = train_df[c].fillna(0.0).astype(np.float32)
        test_df[c] = test_df[c].fillna(0.0).astype(np.float32)
    train_df = train_df.sort_values("event_ts").reset_index(drop=True)
    log(f"Features: {len(feature_cols)}, Cat: {len(cat_cols)}, FB: {len(fb_cols)}")

    # Разбиение на валидацию
    val_window = (train_df["event_ts"] >= VAL_WINDOW_START) & (train_df["event_ts"] < VAL_WINDOW_END)
    val_cands = train_df.loc[val_window, ["customer_id", "event_ts"]].copy()
    val_cands["date"] = val_cands["event_ts"].dt.date
    np.random.seed(SEED)
    cd = val_cands.groupby("customer_id")["date"].apply(lambda x: list(x.unique())).reset_index()
    cd.columns = ["customer_id", "dates"]
    cd["rd"] = cd["dates"].apply(lambda ds: ds[np.random.randint(len(ds))])
    rdm = dict(zip(cd["customer_id"], cd["rd"]))
    val_mask = (train_df["event_ts"].dt.date == train_df["customer_id"].map(rdm)) & val_window
    train_mask = train_df["event_ts"] < VAL_WINDOW_START
    log(f"Val: {val_mask.sum():,} events, {int(train_df.loc[val_mask, 'target_bin'].sum())} red")

    # ── CatBoost MAIN ──
    log("TRAIN: CatBoost MAIN")
    raw = train_df["train_target_raw"].values
    y_main = train_df["target_bin"].astype(np.int8).values
    w = make_weights(raw)
    params_main = {"iterations": 5000, "learning_rate": 0.05, "depth": 8, "od_type": "Iter", "od_wait": 300}
    m_main, bi_main, ap_main, up_main = fit_cb(
        train_df.loc[train_mask, feature_cols], y_main[train_mask], w[train_mask],
        train_df.loc[val_mask, feature_cols], y_main[val_mask], w[val_mask],
        cat_cols, params_main, USE_GPU)

    # ── CatBoost RECENT ──
    log("TRAIN: CatBoost RECENT")
    recent_mask = (train_df["event_ts"] >= RECENT_BORDER) | (raw != -1)
    params_rec = {"iterations": 5000, "learning_rate": 0.05, "depth": 8, "od_type": "Iter", "od_wait": 300}
    m_rec, bi_rec, ap_rec, up_rec = fit_cb(
        train_df.loc[recent_mask & train_mask, feature_cols], y_main[recent_mask & train_mask], w[recent_mask & train_mask],
        train_df.loc[recent_mask & val_mask, feature_cols], y_main[recent_mask & val_mask], w[recent_mask & val_mask],
        cat_cols, params_rec, USE_GPU)

    # ── CatBoost SUSPICIOUS ──
    log("TRAIN: CatBoost SUSPICIOUS")
    y_susp = (raw != -1).astype(np.int8)
    w_susp = np.where(raw != -1, 6.0, 1.2).astype(np.float32)
    params_susp = {"iterations": 5000, "learning_rate": 0.05, "depth": 8, "od_type": "Iter", "od_wait": 300}
    m_susp, bi_susp, _, up_susp = fit_cb(
        train_df.loc[train_mask, feature_cols], y_susp[train_mask], w_susp[train_mask],
        train_df.loc[val_mask, feature_cols], y_susp[val_mask], w_susp[val_mask],
        cat_cols, params_susp, USE_GPU)

    # ── CatBoost RED|SUSP ──
    log("TRAIN: CatBoost RED|SUSP")
    labeled_mask = raw != -1
    rg_train = labeled_mask & train_mask.values
    rg_val = labeled_mask & val_mask.values
    params_rg = {"iterations": 5000, "learning_rate": 0.01, "depth": 4, "l2_leaf_reg": 5, "od_type": "Iter", "od_wait": 500, "eval_metric": "Logloss"}
    w_rg_tr = np.where(raw[rg_train] == 1, 2.2, 1.0).astype(np.float32)
    w_rg_val = np.where(raw[rg_val] == 1, 2.2, 1.0).astype(np.float32)
    m_rg, bi_rg, _, up_rg = fit_cb(
        train_df.loc[rg_train, feature_cols], y_main[rg_train], w_rg_tr,
        train_df.loc[rg_val, feature_cols], y_main[rg_val], w_rg_val,
        cat_cols, params_rg, USE_GPU)

    # ── Блендинг ──
    log("BLEND")
    vpool = Pool(train_df.loc[val_mask, feature_cols], cat_features=cat_cols)
    pv_main = m_main.predict(vpool, prediction_type="RawFormulaVal")
    pv_rec = m_rec.predict(vpool, prediction_type="RawFormulaVal")
    pv_susp = m_susp.predict(vpool, prediction_type="RawFormulaVal")
    pv_rg = m_rg.predict(vpool, prediction_type="RawFormulaVal")
    pv_prod = _logit(_sigmoid(pv_susp) * _sigmoid(pv_rg))
    yv = y_main[val_mask]

    log(f"  Main:    {average_precision_score(yv, pv_main):.6f}")
    log(f"  Recent:  {average_precision_score(yv, pv_rec):.6f}")
    log(f"  Product: {average_precision_score(yv, pv_prod):.6f}")


    best_ap, best_w = -1, None
    for wm in np.arange(0, 0.91, 0.05):
        for wr in np.arange(0, 0.41, 0.05):
            wp = round(1 - wm - wr, 2)
            if wp < 0: continue
            bl = wm * pv_main + wr * pv_rec + wp * pv_prod
            ap = average_precision_score(yv, bl)
            if ap > best_ap: best_ap, best_w = ap, (float(wm), float(wr), float(wp))
    log(f"  CB blend: {best_w}, PR-AUC={best_ap:.6f}")

    cb_bl = best_w[0] * pv_main + best_w[1] * pv_rec + best_w[2] * pv_prod

    # ── CatBoost FEEDBACK (использует FB-фичи) ──
    log("TRAIN: CatBoost FEEDBACK")
    params_fb = {"iterations": 5000, "learning_rate": 0.05, "depth": 8, "od_type": "Iter", "od_wait": 300}
    m_fb, bi_fb, ap_fb, up_fb = fit_cb(
        train_df.loc[train_mask, fb_feature_cols], y_main[train_mask], w[train_mask],
        train_df.loc[val_mask, fb_feature_cols], y_main[val_mask], w[val_mask],
        cat_cols, params_fb, USE_GPU)

    # Предсказание FB-модели на валидации
    fb_vpool = Pool(train_df.loc[val_mask, fb_feature_cols], cat_features=cat_cols)
    pv_fb = m_fb.predict(fb_vpool, prediction_type="RawFormulaVal")
    log(f"  FB model: {average_precision_score(yv, pv_fb):.6f}")

    # Определяем клиентов с историей меток для условной инъекции
    labeled_custs = set(train_df.loc[train_df["train_target_raw"] != -1, "customer_id"].unique())
    val_has_hist = train_df.loc[val_mask, "customer_id"].isin(labeled_custs).values
    log(f"  Val has_hist: {val_has_hist.sum()}/{len(val_has_hist)} ({val_has_hist.mean()*100:.1f}%)")

    # Подбор alpha для инъекции FB на валидации
    r_fb = rankdata(pv_fb) / len(pv_fb)
    r_cb = rankdata(cb_bl) / len(cb_bl)
    best_alpha, best_fb_ap = 0.0, best_ap
    for alpha_cand in np.arange(0.0, 0.51, 0.05):
        blend_fb = r_cb.copy()
        blend_fb[val_has_hist] = (1 - alpha_cand) * r_cb[val_has_hist] + alpha_cand * r_fb[val_has_hist]
        ap = average_precision_score(yv, blend_fb)
        log(f"    alpha={alpha_cand:.2f} -> PR-AUC={ap:.6f}")
        if ap > best_fb_ap: best_fb_ap, best_alpha = ap, float(alpha_cand)
    log(f"  Best alpha={best_alpha:.2f}, PR-AUC={best_fb_ap:.6f} (was {best_ap:.6f})")


    # ── Сохранение результатов для run_coles_refit.py ──
    log("SAVING VAL RESULTS")
    log(f"  CB blend weights: {best_w}")
    log(f"  FB alpha: {best_alpha:.2f}")
    log(f"  Val PR-AUC (CB): {best_ap:.6f}")
    log(f"  Val PR-AUC (CB+FB): {best_fb_ap:.6f}")

    # Сохранение фичей в parquet для рефита
    log("SAVING FEATURES CACHE")
    features_path = CACHE_DIR / "features_all.parquet"
    train_df.to_parquet(features_path, index=False)
    test_df.to_parquet(CACHE_DIR / "test_features.parquet", index=False)
    log(f"  Saved train: {train_df.shape}, test: {test_df.shape}")

    # Сохранение конфига для рефита
    import json
    config = {
        "best_w": list(best_w),
        "best_alpha": best_alpha,
        "feature_cols": feature_cols,
        "fb_feature_cols": fb_feature_cols,
        "cat_cols": cat_cols,
        "bi_main": bi_main, "bi_rec": bi_rec, "bi_susp": bi_susp,
        "bi_rg": bi_rg, "bi_fb": bi_fb,
    }
    with open(CACHE_DIR / "v5_config.json", "w") as f:
        json.dump(config, f)
    log(f"  Config saved to {CACHE_DIR / 'v5_config.json'}")

    log(f"\nTOTAL TIME: {(time.time()-t0)/60:.1f} min")
    log("DONE! → Run run_coles_refit.py for REFIT + test predictions")

if __name__ == "__main__":
    main()

