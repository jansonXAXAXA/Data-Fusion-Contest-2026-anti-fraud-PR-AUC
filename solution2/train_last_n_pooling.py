#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import gc
import json
import logging
import math
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dateutil.relativedelta import relativedelta
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Defaults
# ============================================================

TRAIN_DATASET_ID = 1
TEST_DATASET_ID = 3

TRAIN_START = datetime(2024, 10, 1)
VALIDATION_START = datetime(2025, 1, 1)
TRAIN_END_EXCLUSIVE = datetime(2025, 6, 1)

DEFAULT_HISTORY_WINDOWS_STR = "3,10,30,100"
DEFAULT_FUTURE_WINDOWS_STR = "3,10,30,100"

DEFAULT_LAST_N = 128
DEFAULT_MAX_PRED_SEQ_LEN = 256
DEFAULT_MAX_EPOCHS = 8
DEFAULT_PATIENCE = 3
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.15
DEFAULT_HIDDEN_DIM = 128
DEFAULT_EVENT_DIM = 32
DEFAULT_SLIDING_TRAIN_MONTHS = 3
DEFAULT_BATCH_SIZE_CPU = 16
DEFAULT_BATCH_SIZE_GPU = 64
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_TRAIN_AP_EVERY_N_EPOCHS = 3

DEFAULT_AUX_LOSS_WEIGHT_SUSPICIOUS = 0.5
DEFAULT_AUX_LOSS_WEIGHT_RED_YELLOW = 0.5
DEFAULT_MULTITASK_INFERENCE_BLEND = 0.4

DEFAULT_SESSION_BRANCH_WEIGHT = 0.5
DEFAULT_ONE_HOT_MAX_CARDINALITY = 0  # 0 = disabled

DEFAULT_FUTURE_MAX_HOURS = 24.0
DEFAULT_FUTURE_BRANCH_WEIGHT = 0.5


BASE_CAT_COLS = [
    "event_desc",
    "event_type_nm",
    "channel_indicator_type",
    "channel_indicator_sub_type",
    "mcc_code",
    "pos_cd",
    "currency_iso_cd",
    "browser_language",
    "timezone",
    "operating_system_type",
    "developer_tools",
    "phone_voip_call_state",
    "web_rdp_connection",
    "compromised",
    "accept_lang_primary",
]

BASE_NUM_COLS = [
    "amount_log",
    "amount_missing",
    "battery_value",
    "battery_available",
    "battery_neg",
    "device_version_norm",
    "device_parts_norm",
    "screen1_norm",
    "screen2_norm",
    "session_present",
    "delta_time_log",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "accept_lang_missing",
    "browser_language_known",
    "operating_system_type_known",
    "developer_tools_known",
    "phone_voip_call_state_known",
    "web_rdp_connection_known",
    "compromised_known",
    "device_version_known",
    "screen_known",
    "accept_lang_primary_known",
    "telemetry_available_any",
]

LABEL_HISTORY_NUM_COLS = [
    "lh_prev_yellow_count_log",
    "lh_prev_red_count_log",
    "lh_since_last_yellow_log",
    "lh_since_last_red_log",
    "lh_prev_suspicious_count_log",
    "lh_since_last_suspicious_log",
]

STABLE_CAT_COLS_BASE = [
    "event_desc",
    "event_type_nm",
    "channel_indicator_type",
    "channel_indicator_sub_type",
    "mcc_code",
    "pos_cd",
    "currency_iso_cd",
    "timezone",
]

TELEMETRY_CAT_COLS_BASE = [
    "browser_language",
    "operating_system_type",
    "developer_tools",
    "phone_voip_call_state",
    "web_rdp_connection",
    "compromised",
    "accept_lang_primary",
]

STABLE_NUM_COLS_BASE = [
    "amount_log",
    "amount_missing",
    "delta_time_log",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]

TELEMETRY_NUM_COLS_BASE = [
    "battery_value",
    "battery_available",
    "battery_neg",
    "device_version_norm",
    "device_parts_norm",
    "screen1_norm",
    "screen2_norm",
    "session_present",
    "accept_lang_missing",
    "browser_language_known",
    "operating_system_type_known",
    "developer_tools_known",
    "phone_voip_call_state_known",
    "web_rdp_connection_known",
    "compromised_known",
    "device_version_known",
    "screen_known",
    "accept_lang_primary_known",
    "telemetry_available_any",
]

LABEL_HISTORY_KEY_DTYPE = np.dtype([("ts", np.int64), ("pos", np.int64)])

if set(STABLE_NUM_COLS_BASE).intersection(set(TELEMETRY_NUM_COLS_BASE)):
    raise ValueError("Stable/telemetry numeric column groups must be disjoint.")
if set(STABLE_NUM_COLS_BASE).union(set(TELEMETRY_NUM_COLS_BASE)) != set(BASE_NUM_COLS):
    raise ValueError("Stable/telemetry numeric column groups must cover all BASE_NUM_COLS.")


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ModelConfig:
    history_windows: List[int]
    future_windows: List[int]
    all_num_cols: List[str]
    hidden_dim: int
    event_dim: int
    dropout: float
    multitask_inference_blend: float
    use_session_branch: bool
    session_branch_weight: float
    one_hot_max_cardinality: int
    stable_cat_cols: List[str]
    telemetry_cat_cols: List[str]
    stable_num_cols: List[str]
    telemetry_num_cols: List[str]
    label_history_num_cols: List[str]
    use_future_branches: bool
    future_max_hours: float
    future_branch_weight: float


@dataclass
class SequenceStore:
    event_id: np.ndarray
    customer_id: np.ndarray
    event_dttm: np.ndarray
    dataset_id: np.ndarray
    target_raw: np.ndarray
    session_id_value: np.ndarray
    telemetry_hist_available: np.ndarray
    active_cat_cols: List[str]
    active_num_cols: List[str]
    raw_cat_arrays: Dict[str, np.ndarray]
    num_matrix: np.ndarray


@dataclass
class EncodedSequenceStore:
    event_id: np.ndarray
    customer_id: np.ndarray
    event_dttm: np.ndarray
    dataset_id: np.ndarray
    target_raw: np.ndarray
    session_id_value: np.ndarray
    telemetry_hist_available: np.ndarray
    active_cat_cols: List[str]
    active_num_cols: List[str]
    cat_matrix: np.ndarray
    num_matrix: np.ndarray


# ============================================================
# CLI / utils
# ============================================================

def parse_int_list(arg: str) -> List[int]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty list.")
    vals = []
    for p in parts:
        v = int(p)
        if v <= 0:
            raise ValueError("All windows must be > 0.")
        vals.append(v)
    vals = sorted(set(vals))
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simplified two-branch multi-window sequence pooling model "
            "(mean pooling only; optional same-session branch; optional future branches; "
            "optional explicit label-history) with temporal CV."
        )
    )

    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cv-mode", required=True, choices=["expanding", "sliding"])
    parser.add_argument("--train-final-model", action="store_true")

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--gpu", action="store_true")

    parser.add_argument(
        "--history-windows",
        type=str,
        default=DEFAULT_HISTORY_WINDOWS_STR,
        help="Comma-separated list of past-history windows, e.g. 3,10,30,100.",
    )
    parser.add_argument(
        "--future-windows",
        type=str,
        default=DEFAULT_FUTURE_WINDOWS_STR,
        help="Comma-separated list of future-feature windows, e.g. 3,10,30,100.",
    )

    parser.add_argument("--last-n", type=int, default=DEFAULT_LAST_N)
    parser.add_argument("--max-pred-seq-len", type=int, default=DEFAULT_MAX_PRED_SEQ_LEN)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--event-dim", type=int, default=DEFAULT_EVENT_DIM)
    parser.add_argument("--sliding-train-months", type=int, default=DEFAULT_SLIDING_TRAIN_MONTHS)
    parser.add_argument("--batch-size-cpu", type=int, default=DEFAULT_BATCH_SIZE_CPU)
    parser.add_argument("--batch-size-gpu", type=int, default=DEFAULT_BATCH_SIZE_GPU)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--train-ap-every-n-epochs", type=int, default=DEFAULT_TRAIN_AP_EVERY_N_EPOCHS)

    parser.add_argument(
        "--aux-loss-weight-suspicious",
        type=float,
        default=DEFAULT_AUX_LOSS_WEIGHT_SUSPICIOUS,
    )
    parser.add_argument(
        "--aux-loss-weight-red-yellow",
        type=float,
        default=DEFAULT_AUX_LOSS_WEIGHT_RED_YELLOW,
    )
    parser.add_argument(
        "--multitask-inference-blend",
        type=float,
        default=DEFAULT_MULTITASK_INFERENCE_BLEND,
    )

    parser.add_argument("--use-session-branch", action="store_true")
    parser.add_argument(
        "--session-branch-weight",
        type=float,
        default=DEFAULT_SESSION_BRANCH_WEIGHT,
    )

    parser.add_argument(
        "--use-future-branches",
        action="store_true",
        help="Enable future raw-event branches using later events within a limited horizon.",
    )
    parser.add_argument(
        "--future-max-hours",
        type=float,
        default=DEFAULT_FUTURE_MAX_HOURS,
        help="Maximum future horizon in hours for future branches.",
    )
    parser.add_argument(
        "--future-branch-weight",
        type=float,
        default=DEFAULT_FUTURE_BRANCH_WEIGHT,
        help="Weight multiplier for future branch pooled features.",
    )

    parser.add_argument(
        "--one-hot-max-cardinality",
        type=int,
        default=DEFAULT_ONE_HOT_MAX_CARDINALITY,
    )

    parser.add_argument(
        "--target1-sample-frac",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--use-label-history",
        action="store_true",
        help="Add explicit causal label-history features based on past yellow/red events.",
    )

    args = parser.parse_args()
    args.history_windows = parse_int_list(args.history_windows)
    args.future_windows = parse_int_list(args.future_windows)

    if not (0.0 < args.target1_sample_frac <= 1.0):
        raise ValueError("--target1-sample-frac must be in (0, 1].")
    if args.last_n <= 0:
        raise ValueError("--last-n must be > 0.")
    if args.last_n < max(args.history_windows):
        raise ValueError(f"--last-n must be >= max(history_windows)={max(args.history_windows)}.")
    if args.max_pred_seq_len <= 0:
        raise ValueError("--max-pred-seq-len must be > 0.")
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be > 0.")
    if args.patience <= 0:
        raise ValueError("--patience must be > 0.")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0.")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be in [0, 1).")
    if args.hidden_dim <= 0 or args.event_dim <= 0:
        raise ValueError("--hidden-dim and --event-dim must be > 0.")
    if args.sliding_train_months <= 0:
        raise ValueError("--sliding-train-months must be > 0.")
    if args.batch_size_cpu <= 0 or args.batch_size_gpu <= 0:
        raise ValueError("--batch-size-cpu and --batch-size-gpu must be > 0.")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be > 0.")
    if args.train_ap_every_n_epochs <= 0:
        raise ValueError("--train-ap-every-n-epochs must be > 0.")
    if args.aux_loss_weight_suspicious < 0:
        raise ValueError("--aux-loss-weight-suspicious must be >= 0.")
    if args.aux_loss_weight_red_yellow < 0:
        raise ValueError("--aux-loss-weight-red-yellow must be >= 0.")
    if not (0.0 <= args.multitask_inference_blend <= 1.0):
        raise ValueError("--multitask-inference-blend must be in [0, 1].")
    if args.session_branch_weight < 0:
        raise ValueError("--session-branch-weight must be >= 0.")
    if args.one_hot_max_cardinality < 0:
        raise ValueError("--one-hot-max-cardinality must be >= 0.")
    if args.future_max_hours <= 0:
        raise ValueError("--future-max-hours must be > 0.")
    if args.future_branch_weight < 0:
        raise ValueError("--future-branch-weight must be >= 0.")

    return args


def get_active_cat_cols() -> List[str]:
    return list(BASE_CAT_COLS)


def get_active_num_cols(use_label_history: bool) -> List[str]:
    cols = list(BASE_NUM_COLS)
    if use_label_history:
        cols.extend(LABEL_HISTORY_NUM_COLS)
    return cols


def get_branch_num_cols() -> Tuple[List[str], List[str]]:
    stable = list(STABLE_NUM_COLS_BASE)
    telemetry = list(TELEMETRY_NUM_COLS_BASE)
    if set(stable).intersection(set(telemetry)):
        raise ValueError("Stable/telemetry numeric column groups must be disjoint.")
    return stable, telemetry


def get_branch_cat_cols(active_cat_cols: List[str]) -> Tuple[List[str], List[str]]:
    stable_set = set(STABLE_CAT_COLS_BASE)
    telemetry_set = set(TELEMETRY_CAT_COLS_BASE)

    stable = [c for c in active_cat_cols if c in stable_set]
    telemetry = [c for c in active_cat_cols if c in telemetry_set]
    unknown = [c for c in active_cat_cols if c not in stable_set and c not in telemetry_set]
    if unknown:
        raise ValueError(f"Unassigned categorical columns found: {unknown}")
    return stable, telemetry


def build_model_config(
    args: argparse.Namespace,
    active_cat_cols: List[str],
    active_num_cols: List[str],
) -> ModelConfig:
    stable_cat_cols, telemetry_cat_cols = get_branch_cat_cols(active_cat_cols)
    stable_num_cols, telemetry_num_cols = get_branch_num_cols()

    return ModelConfig(
        history_windows=list(args.history_windows),
        future_windows=list(args.future_windows),
        all_num_cols=list(active_num_cols),
        hidden_dim=args.hidden_dim,
        event_dim=args.event_dim,
        dropout=args.dropout,
        multitask_inference_blend=args.multitask_inference_blend,
        use_session_branch=args.use_session_branch,
        session_branch_weight=args.session_branch_weight,
        one_hot_max_cardinality=args.one_hot_max_cardinality,
        stable_cat_cols=stable_cat_cols,
        telemetry_cat_cols=telemetry_cat_cols,
        stable_num_cols=stable_num_cols,
        telemetry_num_cols=telemetry_num_cols,
        label_history_num_cols=list(LABEL_HISTORY_NUM_COLS if args.use_label_history else []),
        use_future_branches=bool(args.use_future_branches),
        future_max_hours=float(args.future_max_hours),
        future_branch_weight=float(args.future_branch_weight),
    )


def resolve_output_dir(pattern: str) -> Path:
    now = datetime.now()
    resolved = pattern.format(
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H-%M-%S"),
    )
    path = Path(resolved)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("two_branch_multi_window_pooling_simplified_refactored")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(output_dir / "debug.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return str(obj)
    return str(obj)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    positives = int(np.sum(y_true))
    if positives == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def mean_ignore_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def embedding_dim(cardinality_with_padding: int) -> int:
    return int(min(64, max(4, round(1.6 * (cardinality_with_padding ** 0.35)))))


def dt64(x: datetime) -> np.datetime64:
    return np.datetime64(x)


def map_positions(sorted_index: np.ndarray, query_index: np.ndarray) -> np.ndarray:
    if len(query_index) == 0:
        return np.empty(0, dtype=np.int64)
    pos = np.searchsorted(sorted_index, query_index)
    if np.any(pos >= len(sorted_index)) or np.any(sorted_index[pos] != query_index):
        raise ValueError("Index mapping failed.")
    return pos


def pick_keys(d: dict, keys: List[str]) -> dict:
    return {k: d[k] for k in keys}


def save_event_prediction_files(
    event_id: np.ndarray,
    predict: np.ndarray,
    parquet_path: Path,
    csv_path: Optional[Path] = None,
) -> None:
    df = pl.DataFrame(
        {
            "event_id": event_id,
            "predict": predict,
        }
    )
    df.write_parquet(parquet_path)
    if csv_path is not None:
        df.write_csv(csv_path)


def make_monthly_folds(
    cv_mode: str,
    sliding_train_months: int,
) -> List[dict]:
    folds = []
    valid_start = VALIDATION_START
    fold_idx = 0

    while valid_start < TRAIN_END_EXCLUSIVE:
        valid_end = valid_start + relativedelta(months=1)
        if valid_end > TRAIN_END_EXCLUSIVE:
            valid_end = TRAIN_END_EXCLUSIVE

        train_end_exclusive = valid_start

        if cv_mode == "expanding":
            train_start = TRAIN_START
        else:
            train_start = max(
                TRAIN_START,
                train_end_exclusive - relativedelta(months=sliding_train_months),
            )

        if train_end_exclusive <= train_start:
            valid_start = valid_end
            continue

        folds.append(
            {
                "fold_idx": fold_idx,
                "train_start": train_start,
                "train_end_exclusive": train_end_exclusive,
                "valid_start": valid_start,
                "valid_end_exclusive": valid_end,
            }
        )
        fold_idx += 1
        valid_start = valid_end

    return folds


def sample_target1_mask(
    base_mask: np.ndarray,
    target_raw: np.ndarray,
    frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = base_mask.copy()
    if frac >= 1.0:
        return sampled

    target1_mask = base_mask & (target_raw == 1)
    idx = np.flatnonzero(target1_mask)
    if len(idx) == 0:
        return sampled

    keep = rng.random(len(idx)) < frac
    sampled[idx] = keep
    return sampled


def compute_pos_weight_from_binary(binary_targets: np.ndarray, max_cap: float = 100.0) -> float:
    pos = float(np.sum(binary_targets == 1))
    neg = float(np.sum(binary_targets == 0))
    if pos <= 0:
        return 1.0
    return float(min(max_cap, max(1.0, neg / pos)))


def build_cat_mode_info(
    cat_cols: List[str],
    cat_real_cardinalities: Dict[str, int],
    one_hot_max_cardinality: int,
) -> Dict[str, str]:
    modes = {}
    for c in cat_cols:
        real_card = cat_real_cardinalities[c]
        if one_hot_max_cardinality > 0 and real_card <= one_hot_max_cardinality:
            modes[c] = "one_hot"
        else:
            modes[c] = "embedding"
    return modes


def fmt_optional(x: Optional[float]) -> str:
    if x is None:
        return "none"
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    return f"{x:.6f}"


def compute_ap_metrics(
    target_raw: np.ndarray,
    final_score: np.ndarray,
    red_score: np.ndarray,
    suspicious_score: np.ndarray,
    ry_score: np.ndarray,
) -> dict:
    labeled_mask = target_raw > 0

    if np.any(labeled_mask):
        y_red_all = (target_raw[labeled_mask] == 3).astype(np.float32)
        y_susp_all = np.isin(target_raw[labeled_mask], [2, 3]).astype(np.float32)

        final_ap = safe_average_precision(y_red_all, final_score[labeled_mask])
        red_ap = safe_average_precision(y_red_all, red_score[labeled_mask])
        suspicious_ap = safe_average_precision(y_susp_all, suspicious_score[labeled_mask])
    else:
        final_ap = float("nan")
        red_ap = float("nan")
        suspicious_ap = float("nan")

    ry_mask = np.isin(target_raw, [2, 3])
    ry_ap = safe_average_precision(
        (target_raw[ry_mask] == 3).astype(np.float32),
        ry_score[ry_mask],
    ) if np.any(ry_mask) else float("nan")

    return {
        "final_ap": None if np.isnan(final_ap) else float(final_ap),
        "red_ap": None if np.isnan(red_ap) else float(red_ap),
        "suspicious_ap": None if np.isnan(suspicious_ap) else float(suspicious_ap),
        "ry_ap": None if np.isnan(ry_ap) else float(ry_ap),
    }


def build_ensemble_prediction(pred_arrays: List[np.ndarray]) -> np.ndarray:
    if not pred_arrays:
        return np.empty(0, dtype=np.float32)
    stack = np.vstack(pred_arrays)
    with np.errstate(invalid="ignore"):
        pred = np.nanmean(stack, axis=0)
    return pred.astype(np.float32)


def encode_with_unk(
    values: np.ndarray,
    fit_values: np.ndarray,
) -> Tuple[np.ndarray, int, int, float, np.ndarray]:
    codes = np.ones(len(values), dtype=np.int64)

    uniq = np.unique(fit_values)
    if len(uniq) == 0:
        total_cardinality = 2
        real_cardinality = 1
        unk_rate = 1.0
        return codes, total_cardinality, real_cardinality, unk_rate, uniq

    pos = np.searchsorted(uniq, values)
    valid = pos < len(uniq)
    known = np.zeros(len(values), dtype=np.bool_)
    if np.any(valid):
        valid_idx = np.flatnonzero(valid)
        known_valid = uniq[pos[valid]] == values[valid]
        known[valid_idx] = known_valid

    codes[known] = pos[known].astype(np.int64) + 2

    total_cardinality = int(len(uniq) + 2)
    real_cardinality = int(len(uniq) + 1)
    unk_rate = float(np.mean(codes == 1))
    return codes, total_cardinality, real_cardinality, unk_rate, uniq


def _fill_count_since_features(
    ts_seg_ns: np.ndarray,
    query_keys: np.ndarray,
    visible_pos_idx: np.ndarray,
    out_count_col: np.ndarray,
    out_since_col: np.ndarray,
    default_time_log: float,
) -> None:
    if len(visible_pos_idx) == 0:
        out_count_col.fill(0.0)
        out_since_col.fill(default_time_log)
        return

    hist_keys = np.empty(len(visible_pos_idx), dtype=LABEL_HISTORY_KEY_DTYPE)
    hist_keys["ts"] = ts_seg_ns[visible_pos_idx]
    hist_keys["pos"] = visible_pos_idx

    counts = np.searchsorted(hist_keys, query_keys, side="right").astype(np.int64, copy=False)

    out_count_col[:] = np.log1p(counts).astype(np.float32)
    out_since_col.fill(default_time_log)

    has_prev = counts > 0
    if np.any(has_prev):
        prev_pos = visible_pos_idx[counts[has_prev] - 1]
        delta_ns = ts_seg_ns[has_prev] - ts_seg_ns[prev_pos]
        delta_sec = np.clip(
            delta_ns.astype(np.float64) / 1_000_000_000.0,
            0.0,
            365.0 * 24.0 * 3600.0,
        )
        out_since_col[has_prev] = np.log1p(delta_sec).astype(np.float32)


def materialize_label_history_features(
    customer_id: np.ndarray,
    event_dttm: np.ndarray,
    target_raw: np.ndarray,
    logger: Optional[logging.Logger] = None,
    tag: str = "",
) -> Tuple[np.ndarray, dict]:
    if logger is not None:
        logger.info("%s materializing no-lag causal label-history features ...", tag)

    idx_y_count = LABEL_HISTORY_NUM_COLS.index("lh_prev_yellow_count_log")
    idx_r_count = LABEL_HISTORY_NUM_COLS.index("lh_prev_red_count_log")
    idx_y_time = LABEL_HISTORY_NUM_COLS.index("lh_since_last_yellow_log")
    idx_r_time = LABEL_HISTORY_NUM_COLS.index("lh_since_last_red_log")
    idx_s_count = LABEL_HISTORY_NUM_COLS.index("lh_prev_suspicious_count_log")
    idx_s_time = LABEL_HISTORY_NUM_COLS.index("lh_since_last_suspicious_log")

    ts_ns = event_dttm.astype("datetime64[ns]").astype(np.int64)
    default_time_log = float(np.log1p(365.0 * 24.0 * 3600.0))

    out = np.zeros((len(customer_id), len(LABEL_HISTORY_NUM_COLS)), dtype=np.float32)

    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.flatnonzero(customer_id[1:] != customer_id[:-1]).astype(np.int64) + 1,
            np.array([len(customer_id)], dtype=np.int64),
        ]
    )

    n_customers = len(boundaries) - 1
    for g, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if logger is not None and g > 0 and g % 10000 == 0:
            logger.info("%s label-history progress: processed %d customers / %d", tag, g, n_customers)

        target_seg = target_raw[start:end]
        ts_seg_ns = ts_ns[start:end]

        m = end - start
        rel = np.arange(m, dtype=np.int64)

        query_keys = np.empty(m, dtype=LABEL_HISTORY_KEY_DTYPE)
        query_keys["ts"] = ts_seg_ns
        query_keys["pos"] = rel - 1

        pos_y = np.flatnonzero(target_seg == 2)
        pos_r = np.flatnonzero(target_seg == 3)
        pos_s = np.flatnonzero((target_seg == 2) | (target_seg == 3))

        _fill_count_since_features(
            ts_seg_ns=ts_seg_ns,
            query_keys=query_keys,
            visible_pos_idx=pos_y,
            out_count_col=out[start:end, idx_y_count],
            out_since_col=out[start:end, idx_y_time],
            default_time_log=default_time_log,
        )
        _fill_count_since_features(
            ts_seg_ns=ts_seg_ns,
            query_keys=query_keys,
            visible_pos_idx=pos_r,
            out_count_col=out[start:end, idx_r_count],
            out_since_col=out[start:end, idx_r_time],
            default_time_log=default_time_log,
        )
        _fill_count_since_features(
            ts_seg_ns=ts_seg_ns,
            query_keys=query_keys,
            visible_pos_idx=pos_s,
            out_count_col=out[start:end, idx_s_count],
            out_since_col=out[start:end, idx_s_time],
            default_time_log=default_time_log,
        )

    stats = {
        "label_history_enabled": True,
        "lag_days": 0,
        "hidden_intervals": [],
    }
    return out, stats


# ============================================================
# Loading / preprocessing
# ============================================================

def load_and_preprocess(
    input_path: str,
    active_cat_cols: List[str],
    use_label_history: bool,
    logger: logging.Logger,
) -> Tuple[SequenceStore, Optional[dict]]:
    active_num_cols = get_active_num_cols(use_label_history=use_label_history)

    logger.info("Loading parquet with polars: %s", input_path)
    logger.info("Assumption: input is sorted by customer_id, event_dttm.")
    logger.info("Active categorical columns: %s", active_cat_cols)
    logger.info("Active numerical columns: %s", active_num_cols)

    raw_cols = [
        "customer_id",
        "event_id",
        "event_dttm",
        "dataset",
        "target",
        "event_desc",
        "event_type_nm",
        "channel_indicator_type",
        "channel_indicator_sub_type",
        "currency_iso_cd",
        "mcc_code",
        "pos_cd",
        "accept_language",
        "browser_language",
        "timezone",
        "operating_system_type",
        "developer_tools",
        "phone_voip_call_state",
        "web_rdp_connection",
        "compromised",
        "operaton_amt",
        "session_id",
        "battery",
        "device_system_version",
        "device_system_version_parts",
        "screen_size_1",
        "screen_size_2",
    ]

    dt_ns = pl.col("event_dttm").dt.cast_time_unit("ns").cast(pl.Int64)
    delta_sec_expr = dt_ns.diff().over("customer_id") / 1_000_000_000.0

    hour_expr = pl.col("event_dttm").dt.hour().cast(pl.Float32)
    dow_expr = (pl.col("event_dttm").dt.weekday().cast(pl.Float32) - 1.0)

    amount_clean = (
        pl.when(pl.col("operaton_amt").is_null() | (pl.col("operaton_amt") < 0))
        .then(0.0)
        .otherwise(pl.col("operaton_amt"))
    )

    battery_col = pl.col("battery")
    battery_available_expr = (
        battery_col.fill_null(float("nan")).is_finite() &
        (battery_col.fill_null(float("nan")) >= 0)
    )
    battery_value_expr = (
        pl.when(battery_available_expr)
        .then(battery_col.clip(lower_bound=0.0, upper_bound=100.0) / 100.0)
        .otherwise(0.0)
    )
    battery_neg_expr = (
        pl.when(battery_col.fill_null(float("nan")).is_finite() & (battery_col < 0))
        .then((-battery_col).cast(pl.Float32))
        .otherwise(0.0)
    )

    accept_lang_str = pl.col("accept_language").cast(pl.Utf8).fill_null("").str.to_lowercase()

    accept_lang_primary_str = (
        accept_lang_str
        .str.split(",").list.get(0).fill_null("")
        .str.split(";").list.get(0).fill_null("")
        .str.split("-").list.get(0).fill_null("")
    )

    accept_lang_primary_hash_expr = (
        pl.when(accept_lang_primary_str == "")
        .then(pl.lit(0, dtype=pl.UInt64))
        .otherwise(accept_lang_primary_str.hash().cast(pl.UInt64))
        .alias("accept_lang_primary")
    )

    session_raw_expr = pl.col("session_id").cast(pl.Float64).fill_null(float("nan"))
    session_present_expr = session_raw_expr.is_finite()

    browser_language_known_expr = (pl.col("browser_language").fill_null(0) > 0)
    operating_system_type_known_expr = (pl.col("operating_system_type").fill_null(0) > 0)
    developer_tools_known_expr = (pl.col("developer_tools").fill_null(0) > 0)
    phone_voip_call_state_known_expr = (pl.col("phone_voip_call_state").fill_null(0) > 0)
    web_rdp_connection_known_expr = (pl.col("web_rdp_connection").fill_null(0) > 0)
    compromised_known_expr = (pl.col("compromised").fill_null(0) > 0)
    device_version_known_expr = (pl.col("device_system_version").fill_null(0) > 0)
    screen_known_expr = (
        (pl.col("screen_size_1").fill_null(0) > 0) |
        (pl.col("screen_size_2").fill_null(0) > 0)
    )
    accept_lang_primary_known_expr = (accept_lang_primary_str != "")

    telemetry_available_any_expr = (
        session_present_expr |
        browser_language_known_expr |
        operating_system_type_known_expr |
        developer_tools_known_expr |
        phone_voip_call_state_known_expr |
        web_rdp_connection_known_expr |
        compromised_known_expr |
        battery_available_expr |
        device_version_known_expr |
        screen_known_expr |
        accept_lang_primary_known_expr
    )

    derived_exprs = [
        pl.col("customer_id").cast(pl.Int64),
        pl.col("event_id").cast(pl.Int64),
        pl.col("event_dttm").dt.cast_time_unit("ns"),
        pl.col("dataset").cast(pl.UInt8).alias("dataset"),
        pl.col("target").cast(pl.UInt8).alias("target"),

        pl.col("event_desc").fill_null(0).cast(pl.UInt8).alias("event_desc"),
        pl.col("event_type_nm").fill_null(0).cast(pl.UInt8).alias("event_type_nm"),
        pl.col("channel_indicator_type").fill_null(0).cast(pl.UInt8).alias("channel_indicator_type"),
        pl.col("channel_indicator_sub_type").fill_null(0).cast(pl.UInt8).alias("channel_indicator_sub_type"),
        pl.col("currency_iso_cd").fill_null(0).cast(pl.UInt8).alias("currency_iso_cd"),
        pl.col("mcc_code").fill_null(0).cast(pl.UInt8).alias("mcc_code"),
        pl.col("pos_cd").fill_null(0).cast(pl.UInt8).alias("pos_cd"),
        pl.col("browser_language").fill_null(0).cast(pl.UInt8).alias("browser_language"),
        pl.col("timezone").fill_null(0).cast(pl.UInt16).alias("timezone"),
        pl.col("operating_system_type").fill_null(0).cast(pl.UInt8).alias("operating_system_type"),
        pl.col("developer_tools").fill_null(0).cast(pl.UInt8).alias("developer_tools"),
        pl.col("phone_voip_call_state").fill_null(0).cast(pl.UInt8).alias("phone_voip_call_state"),
        pl.col("web_rdp_connection").fill_null(0).cast(pl.UInt8).alias("web_rdp_connection"),
        pl.col("compromised").fill_null(0).cast(pl.UInt8).alias("compromised"),

        accept_lang_primary_hash_expr,

        amount_clean.log1p().cast(pl.Float32).alias("amount_log"),
        pl.col("operaton_amt").is_null().cast(pl.Float32).alias("amount_missing"),

        battery_value_expr.cast(pl.Float32).alias("battery_value"),
        battery_available_expr.cast(pl.Float32).alias("battery_available"),
        battery_neg_expr.cast(pl.Float32).alias("battery_neg"),

        (pl.col("device_system_version").fill_null(0).cast(pl.Float32) / 100000.0).alias("device_version_norm"),
        (pl.col("device_system_version_parts").fill_null(0).cast(pl.Float32) / 4.0).alias("device_parts_norm"),
        (pl.col("screen_size_1").fill_null(0).cast(pl.Float32) / 4000.0).alias("screen1_norm"),
        (pl.col("screen_size_2").fill_null(0).cast(pl.Float32) / 4000.0).alias("screen2_norm"),
        session_raw_expr.alias("__session_id_raw"),
        session_present_expr.cast(pl.Float32).alias("session_present"),

        (
            pl.when(delta_sec_expr.is_null() | (delta_sec_expr < 0))
            .then(30.0 * 24 * 3600)
            .otherwise(delta_sec_expr.clip(lower_bound=0.0, upper_bound=365.0 * 24 * 3600))
            .log1p()
            .cast(pl.Float32)
        ).alias("delta_time_log"),

        (hour_expr * (2.0 * math.pi / 24.0)).sin().cast(pl.Float32).alias("hour_sin"),
        (hour_expr * (2.0 * math.pi / 24.0)).cos().cast(pl.Float32).alias("hour_cos"),
        (dow_expr * (2.0 * math.pi / 7.0)).sin().cast(pl.Float32).alias("dow_sin"),
        (dow_expr * (2.0 * math.pi / 7.0)).cos().cast(pl.Float32).alias("dow_cos"),
        (dow_expr >= 5.0).cast(pl.Float32).alias("is_weekend"),

        (accept_lang_str == "").cast(pl.Float32).alias("accept_lang_missing"),
        browser_language_known_expr.cast(pl.Float32).alias("browser_language_known"),
        operating_system_type_known_expr.cast(pl.Float32).alias("operating_system_type_known"),
        developer_tools_known_expr.cast(pl.Float32).alias("developer_tools_known"),
        phone_voip_call_state_known_expr.cast(pl.Float32).alias("phone_voip_call_state_known"),
        web_rdp_connection_known_expr.cast(pl.Float32).alias("web_rdp_connection_known"),
        compromised_known_expr.cast(pl.Float32).alias("compromised_known"),
        device_version_known_expr.cast(pl.Float32).alias("device_version_known"),
        screen_known_expr.cast(pl.Float32).alias("screen_known"),
        accept_lang_primary_known_expr.cast(pl.Float32).alias("accept_lang_primary_known"),
        telemetry_available_any_expr.cast(pl.Float32).alias("telemetry_available_any"),
    ]

    collect_cols = (
        ["customer_id", "event_id", "event_dttm", "dataset", "target", "__session_id_raw"]
        + active_cat_cols
        + BASE_NUM_COLS
    )

    lf = (
        pl.scan_parquet(input_path)
        .select(raw_cols)
        .with_columns(derived_exprs)
    )

    df = lf.select(collect_cols).collect()

    logger.info("Loaded dataframe shape: %s", df.shape)

    overview = (
        df.group_by("dataset")
        .agg(
            pl.len().alias("rows"),
            (pl.col("target") == 3).sum().alias("red"),
            (pl.col("target") == 2).sum().alias("yellow"),
            (pl.col("target") == 1).sum().alias("green"),
            pl.col("telemetry_available_any").mean().alias("telemetry_available_any_rate"),
            pl.col("session_present").mean().alias("session_present_rate"),
        )
        .sort("dataset")
    )
    logger.info("Dataset overview:\n%s", overview)

    event_id = df["event_id"].to_numpy()
    customer_id = df["customer_id"].to_numpy()
    event_dttm = df["event_dttm"].to_numpy()
    dataset_id = df["dataset"].to_numpy().astype(np.uint8, copy=False)
    target_raw = df["target"].to_numpy().astype(np.uint8, copy=False)

    session_raw = df["__session_id_raw"].to_numpy().astype(np.float64, copy=False)
    session_present_np = np.isfinite(session_raw)

    session_id_value = np.zeros(len(session_raw), dtype=np.int64)
    if np.any(session_present_np):
        session_id_value[session_present_np] = session_raw[session_present_np].astype(np.int64, copy=False) + 1

    df_session_present = df["session_present"].to_numpy().astype(np.float32, copy=False)
    if not np.array_equal(df_session_present.astype(np.int8, copy=False), session_present_np.astype(np.int8, copy=False)):
        raise ValueError("session_present and session_id_value availability are inconsistent.")

    raw_cat_arrays: Dict[str, np.ndarray] = {}
    for c in active_cat_cols:
        raw_cat_arrays[c] = df[c].to_numpy()

    num_matrix = np.zeros((len(df), len(active_num_cols)), dtype=np.float32)
    base_num_col_index = {c: i for i, c in enumerate(BASE_NUM_COLS)}
    for c in BASE_NUM_COLS:
        num_matrix[:, base_num_col_index[c]] = df[c].to_numpy().astype(np.float32, copy=False)

    label_history_stats = None
    if use_label_history:
        label_history_block, label_history_stats = materialize_label_history_features(
            customer_id=customer_id,
            event_dttm=event_dttm,
            target_raw=target_raw,
            logger=logger,
            tag="Global",
        )
        start = len(BASE_NUM_COLS)
        num_matrix[:, start:start + len(LABEL_HISTORY_NUM_COLS)] = label_history_block
        del label_history_block
        gc.collect()

    telemetry_hist_available = df["telemetry_available_any"].to_numpy().astype(np.bool_, copy=False)

    del df
    gc.collect()

    store = SequenceStore(
        event_id=event_id,
        customer_id=customer_id,
        event_dttm=event_dttm,
        dataset_id=dataset_id,
        target_raw=target_raw,
        session_id_value=session_id_value,
        telemetry_hist_available=telemetry_hist_available,
        active_cat_cols=list(active_cat_cols),
        active_num_cols=list(active_num_cols),
        raw_cat_arrays=raw_cat_arrays,
        num_matrix=num_matrix,
    )
    return store, label_history_stats


def build_encoded_store(
    store: SequenceStore,
    vocab_fit_mask: np.ndarray,
    one_hot_max_cardinality: int,
    logger: Optional[logging.Logger] = None,
    tag: str = "",
) -> Tuple[
    EncodedSequenceStore,
    Dict[str, int],
    Dict[str, int],
    Dict[str, str],
    Dict[str, dict],
    Dict[str, np.ndarray],
]:
    n_rows = len(store.event_id)
    n_cat = len(store.active_cat_cols)

    cat_matrix = np.empty((n_rows, n_cat), dtype=np.int32)

    cat_cardinalities_total: Dict[str, int] = {}
    cat_real_cardinalities: Dict[str, int] = {}
    cat_encoding_stats: Dict[str, dict] = {}
    cat_vocab_values: Dict[str, np.ndarray] = {}

    for i, c in enumerate(store.active_cat_cols):
        values = store.raw_cat_arrays[c]
        fit_values = values[vocab_fit_mask]

        codes, total_card, real_card, unk_rate, vocab_values = encode_with_unk(
            values=values,
            fit_values=fit_values,
        )

        cat_matrix[:, i] = codes.astype(np.int32, copy=False)
        cat_cardinalities_total[c] = total_card
        cat_real_cardinalities[c] = real_card
        cat_vocab_values[c] = vocab_values.copy()

        cat_encoding_stats[c] = {
            "fit_visible_rows": int(np.sum(vocab_fit_mask)),
            "fit_unique_values": int(len(vocab_values)),
            "non_padding_cardinality_including_unk": int(real_card),
            "total_cardinality_with_padding": int(total_card),
            "unk_rate_all_rows": float(unk_rate),
            "unk_rows_all": int(np.sum(codes == 1)),
            "raw_dtype": str(values.dtype),
        }

    cat_mode_info = build_cat_mode_info(
        cat_cols=store.active_cat_cols,
        cat_real_cardinalities=cat_real_cardinalities,
        one_hot_max_cardinality=one_hot_max_cardinality,
    )

    if logger is not None:
        logger.info("%s category encoding built on visible history rows=%d", tag, int(np.sum(vocab_fit_mask)))
        for c in store.active_cat_cols:
            logger.info(
                "%s cat=%s | raw_dtype=%s | mode=%s | fit_unique=%d | non_padding_card=%d | total_card=%d | unk_rate_all=%.6f",
                tag,
                c,
                str(store.raw_cat_arrays[c].dtype),
                cat_mode_info[c],
                cat_encoding_stats[c]["fit_unique_values"],
                cat_real_cardinalities[c],
                cat_cardinalities_total[c],
                cat_encoding_stats[c]["unk_rate_all_rows"],
            )

    encoded_store = EncodedSequenceStore(
        event_id=store.event_id,
        customer_id=store.customer_id,
        event_dttm=store.event_dttm,
        dataset_id=store.dataset_id,
        target_raw=store.target_raw,
        session_id_value=store.session_id_value,
        telemetry_hist_available=store.telemetry_hist_available,
        active_cat_cols=list(store.active_cat_cols),
        active_num_cols=list(store.active_num_cols),
        cat_matrix=cat_matrix,
        num_matrix=store.num_matrix,
    )
    return (
        encoded_store,
        cat_cardinalities_total,
        cat_real_cardinalities,
        cat_mode_info,
        cat_encoding_stats,
        cat_vocab_values,
    )


# ============================================================
# Segments
# ============================================================

def build_segments(
    customer_id: np.ndarray,
    event_dttm: np.ndarray,
    last_n: int,
    max_pred_seq_len: int,
    use_future_branches: bool,
    future_windows: List[int],
    future_max_hours: float,
) -> np.ndarray:
    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.flatnonzero(customer_id[1:] != customer_id[:-1]).astype(np.int64) + 1,
            np.array([len(customer_id)], dtype=np.int64),
        ]
    )

    ts_ns = event_dttm.astype("datetime64[ns]").astype(np.int64)
    future_h_ns = int(round(future_max_hours * 3600.0 * 1_000_000_000.0))
    max_future_window = max(future_windows) if future_windows else 0

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        cust_ts = ts_ns[start:end]
        pos = start

        while pos < end:
            pred_end = min(pos + max_pred_seq_len, end)
            ctx_start = max(start, pos - last_n)

            if use_future_branches and max_future_window > 0:
                last_query_global = pred_end - 1
                last_query_ts = ts_ns[last_query_global]

                count_ctx_end = min(end, pred_end + max_future_window)
                time_ctx_end_local = np.searchsorted(
                    cust_ts,
                    last_query_ts + future_h_ns,
                    side="right",
                )
                time_ctx_end = start + int(time_ctx_end_local)

                ctx_end = min(count_ctx_end, time_ctx_end)
                ctx_end = max(ctx_end, pred_end)
            else:
                ctx_end = pred_end

            segments.append((ctx_start, pos, pred_end, ctx_end))
            pos = pred_end

    return np.asarray(segments, dtype=np.int64)


def filter_segments_by_target_mask(segments: np.ndarray, row_mask: np.ndarray) -> np.ndarray:
    prefix = np.concatenate([[0], np.cumsum(row_mask.astype(np.int64))])
    keep = (prefix[segments[:, 2]] - prefix[segments[:, 1]]) > 0
    return segments[keep]


# ============================================================
# Dataset / collate
# ============================================================

class SequenceSegmentDataset(Dataset):
    def __init__(
        self,
        store: EncodedSequenceStore,
        segments: np.ndarray,
        row_train_mask: np.ndarray,
        row_pred_mask: np.ndarray,
        row_future_visible_mask: Optional[np.ndarray] = None,
    ):
        self.store = store
        self.segments = segments
        self.row_train_mask = row_train_mask
        self.row_pred_mask = row_pred_mask
        self.row_future_visible_mask = (
            row_future_visible_mask
            if row_future_visible_mask is not None
            else np.zeros_like(row_train_mask, dtype=bool)
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict:
        ctx_start, pred_start, pred_end, ctx_end = self.segments[idx]
        seq_slice = slice(ctx_start, ctx_end)

        pred_offset_start = pred_start - ctx_start
        pred_offset_end = pred_end - ctx_start
        seq_len = ctx_end - ctx_start

        cats = self.store.cat_matrix[seq_slice]
        nums = self.store.num_matrix[seq_slice]

        labels = (self.store.target_raw[seq_slice] == 3).astype(np.float32, copy=False)
        target_raw = self.store.target_raw[seq_slice].astype(np.int64, copy=False)
        session_ids = self.store.session_id_value[seq_slice].astype(np.int64, copy=False)
        telemetry_hist_available = self.store.telemetry_hist_available[seq_slice].astype(np.bool_, copy=False)
        row_idx = np.arange(ctx_start, ctx_end, dtype=np.int64)
        event_ts_ns = self.store.event_dttm[seq_slice].astype("datetime64[ns]").astype(np.int64)

        valid_mask = np.ones(seq_len, dtype=np.bool_)
        train_mask = self.row_train_mask[seq_slice].copy()
        pred_mask = self.row_pred_mask[seq_slice].copy()
        future_visible_mask = self.row_future_visible_mask[seq_slice].copy()

        train_mask[:pred_offset_start] = False
        train_mask[pred_offset_end:] = False

        pred_mask[:pred_offset_start] = False
        pred_mask[pred_offset_end:] = False

        return {
            "cats": cats,
            "nums": nums,
            "labels": labels,
            "target_raw": target_raw,
            "session_ids": session_ids,
            "telemetry_hist_available": telemetry_hist_available,
            "valid_mask": valid_mask,
            "train_mask": train_mask,
            "pred_mask": pred_mask,
            "future_visible_mask": future_visible_mask,
            "event_ts_ns": event_ts_ns,
            "row_idx": row_idx,
        }


def collate_segments(batch: List[dict]) -> dict:
    batch_size = len(batch)
    max_len = max(item["cats"].shape[0] for item in batch)
    num_cat = batch[0]["cats"].shape[1]
    num_num = batch[0]["nums"].shape[1]

    cats = torch.zeros((batch_size, max_len, num_cat), dtype=torch.long)
    nums = torch.zeros((batch_size, max_len, num_num), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_len), dtype=torch.float32)
    target_raw = torch.zeros((batch_size, max_len), dtype=torch.long)
    session_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    telemetry_hist_available = torch.zeros((batch_size, max_len), dtype=torch.bool)
    valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    train_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    pred_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    future_visible_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    event_ts_ns = torch.zeros((batch_size, max_len), dtype=torch.long)
    row_idx = torch.full((batch_size, max_len), -1, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["cats"].shape[0]
        cats[i, :L] = torch.as_tensor(item["cats"], dtype=torch.long)
        nums[i, :L] = torch.as_tensor(item["nums"], dtype=torch.float32)
        labels[i, :L] = torch.as_tensor(item["labels"], dtype=torch.float32)
        target_raw[i, :L] = torch.as_tensor(item["target_raw"], dtype=torch.long)
        session_ids[i, :L] = torch.as_tensor(item["session_ids"], dtype=torch.long)
        telemetry_hist_available[i, :L] = torch.as_tensor(item["telemetry_hist_available"], dtype=torch.bool)
        valid_mask[i, :L] = torch.as_tensor(item["valid_mask"], dtype=torch.bool)
        train_mask[i, :L] = torch.as_tensor(item["train_mask"], dtype=torch.bool)
        pred_mask[i, :L] = torch.as_tensor(item["pred_mask"], dtype=torch.bool)
        future_visible_mask[i, :L] = torch.as_tensor(item["future_visible_mask"], dtype=torch.bool)
        event_ts_ns[i, :L] = torch.as_tensor(item["event_ts_ns"], dtype=torch.long)
        row_idx[i, :L] = torch.as_tensor(item["row_idx"], dtype=torch.long)

    return {
        "cats": cats,
        "nums": nums,
        "labels": labels,
        "target_raw": target_raw,
        "session_ids": session_ids,
        "telemetry_hist_available": telemetry_hist_available,
        "valid_mask": valid_mask,
        "train_mask": train_mask,
        "pred_mask": pred_mask,
        "future_visible_mask": future_visible_mask,
        "event_ts_ns": event_ts_ns,
        "row_idx": row_idx,
    }


# ============================================================
# Model
# ============================================================

class MultiTaskTwoBranchMultiWindowModel(nn.Module):
    def __init__(
        self,
        all_cat_cols: List[str],
        cat_cardinalities_total: Dict[str, int],
        cat_real_cardinalities: Dict[str, int],
        model_config: ModelConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.all_cat_cols = list(all_cat_cols)
        self.all_num_cols = list(model_config.all_num_cols)

        self.cat_cardinalities_total = cat_cardinalities_total
        self.cat_real_cardinalities = cat_real_cardinalities

        self.cat_col_index = {c: i for i, c in enumerate(self.all_cat_cols)}
        self.num_col_index = {c: i for i, c in enumerate(self.all_num_cols)}

        self.cat_modes: Dict[str, str] = {}
        self.embeddings = nn.ModuleDict()

        for c in self.all_cat_cols:
            real_card = self.cat_real_cardinalities[c]
            total_card = self.cat_cardinalities_total[c]
            if model_config.one_hot_max_cardinality > 0 and real_card <= model_config.one_hot_max_cardinality:
                self.cat_modes[c] = "one_hot"
            else:
                self.cat_modes[c] = "embedding"
                dim = embedding_dim(total_card)
                self.embeddings[c] = nn.Embedding(total_card, dim, padding_idx=0)

        stable_cat_dim = self._total_cat_feature_dim(model_config.stable_cat_cols)
        telemetry_cat_dim = self._total_cat_feature_dim(model_config.telemetry_cat_cols)

        stable_input_dim = stable_cat_dim + len(model_config.stable_num_cols)
        telemetry_input_dim = telemetry_cat_dim + len(model_config.telemetry_num_cols)

        self.stable_encoder = nn.Sequential(
            nn.Linear(stable_input_dim, model_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim, model_config.event_dim),
            nn.GELU(),
        )

        self.telemetry_encoder = nn.Sequential(
            nn.Linear(telemetry_input_dim, model_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim, model_config.event_dim),
            nn.GELU(),
        )

        per_past_branch_multi = len(model_config.history_windows) * (3 * model_config.event_dim + 1)
        per_future_branch_multi = len(model_config.future_windows) * (3 * model_config.event_dim + 1)

        self.stable_head_dim = model_config.event_dim + per_past_branch_multi
        self.telemetry_head_dim = model_config.event_dim + per_past_branch_multi
        self.session_head_dim = per_past_branch_multi if model_config.use_session_branch else 0
        self.future_stable_head_dim = per_future_branch_multi if model_config.use_future_branches else 0
        self.future_telemetry_head_dim = per_future_branch_multi if model_config.use_future_branches else 0
        self.label_history_head_dim = len(model_config.label_history_num_cols)

        head_in = (
            self.stable_head_dim
            + self.telemetry_head_dim
            + self.session_head_dim
            + self.future_stable_head_dim
            + self.future_telemetry_head_dim
            + self.label_history_head_dim
        )

        self.future_max_ns = int(round(model_config.future_max_hours * 3600.0 * 1_000_000_000.0))

        self.active_branch_names: List[str] = ["stable_past", "telemetry_past"]
        if model_config.use_session_branch:
            self.active_branch_names.append("session")
        if model_config.use_future_branches:
            self.active_branch_names.append("future_stable")
            self.active_branch_names.append("future_telemetry")
        if model_config.label_history_num_cols:
            self.active_branch_names.append("label_history")

        self.head_red = nn.Sequential(
            nn.Linear(head_in, model_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim // 2, 1),
        )
        self.head_suspicious = nn.Sequential(
            nn.Linear(head_in, model_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim // 2, 1),
        )
        self.head_ry = nn.Sequential(
            nn.Linear(head_in, model_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim // 2, 1),
        )

    def _total_cat_feature_dim(self, cols: List[str]) -> int:
        total = 0
        for c in cols:
            if self.model_config.one_hot_max_cardinality > 0 and self.cat_real_cardinalities[c] <= self.model_config.one_hot_max_cardinality:
                total += self.cat_real_cardinalities[c]
            else:
                total += embedding_dim(self.cat_cardinalities_total[c])
        return total

    def _encode_selected_cats(self, cats: torch.Tensor, selected_cols: List[str]) -> torch.Tensor:
        if not selected_cols:
            B, T, _ = cats.shape
            return torch.zeros((B, T, 0), dtype=torch.float32, device=cats.device)

        parts = []
        for c in selected_cols:
            i = self.cat_col_index[c]
            x = cats[:, :, i]
            if self.cat_modes[c] == "one_hot":
                total_card = self.cat_cardinalities_total[c]
                oh = F.one_hot(x, num_classes=total_card).float()
                oh = oh[:, :, 1:]
                parts.append(oh)
            else:
                parts.append(self.embeddings[c](x))
        return torch.cat(parts, dim=-1)

    def _select_num_block(self, nums: torch.Tensor, cols: List[str]) -> torch.Tensor:
        if not cols:
            B, T, _ = nums.shape
            return torch.zeros((B, T, 0), dtype=nums.dtype, device=nums.device)
        idxs = [self.num_col_index[c] for c in cols]
        return nums[:, :, idxs]

    def _encode_branch(
        self,
        cats: torch.Tensor,
        nums: torch.Tensor,
        valid_mask: torch.Tensor,
        cat_cols: List[str],
        num_cols: List[str],
        encoder: nn.Module,
    ) -> torch.Tensor:
        cat_encoded = self._encode_selected_cats(cats, cat_cols)
        num_block = self._select_num_block(nums, num_cols)
        x = torch.cat([cat_encoded, num_block], dim=-1)
        emb = encoder(x)
        emb = emb * valid_mask.unsqueeze(-1).float()
        return emb

    def _make_pair_mask(
        self,
        history_mask: torch.Tensor,
        window: int,
        session_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = history_mask.shape
        t_idx = torch.arange(T, device=history_mask.device).view(T, 1)
        j_idx = torch.arange(T, device=history_mask.device).view(1, T)
        pair_mask = (j_idx < t_idx) & (j_idx >= (t_idx - window))
        pair_mask = pair_mask.unsqueeze(0).expand(B, T, T)
        pair_mask = pair_mask & history_mask.unsqueeze(1)

        if session_ids is not None:
            sid_q = session_ids.unsqueeze(2)
            sid_j = session_ids.unsqueeze(1)
            same = (sid_q == sid_j) & (sid_q != 0) & (sid_j != 0)
            pair_mask = pair_mask & same

        return pair_mask

    def _make_future_pair_mask(
        self,
        future_visible_mask: torch.Tensor,
        event_ts_ns: torch.Tensor,
        window: int,
        session_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = future_visible_mask.shape
        t_idx = torch.arange(T, device=future_visible_mask.device).view(T, 1)
        j_idx = torch.arange(T, device=future_visible_mask.device).view(1, T)

        pair_mask = (j_idx > t_idx) & (j_idx <= (t_idx + window))
        pair_mask = pair_mask.unsqueeze(0).expand(B, T, T)

        dt_ns = event_ts_ns.unsqueeze(1) - event_ts_ns.unsqueeze(2)  # ts_j - ts_t
        pair_mask = pair_mask & (dt_ns > 0)

        if self.future_max_ns > 0:
            pair_mask = pair_mask & (dt_ns <= self.future_max_ns)

        pair_mask = pair_mask & future_visible_mask.unsqueeze(1)

        if session_ids is not None:
            sid_q = session_ids.unsqueeze(2)
            sid_j = session_ids.unsqueeze(1)
            same = (sid_q == sid_j) & (sid_q != 0) & (sid_j != 0)
            pair_mask = pair_mask & same

        return pair_mask

    def _pool_from_pair_mask_mean(
        self,
        event_emb: torch.Tensor,
        query_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        window: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_f = query_mask.float()

        weights = pair_mask.float()
        weighted_sum = torch.bmm(weights, event_emb)
        weight_sum = weights.sum(dim=-1)

        hist = weighted_sum / weight_sum.clamp_min(1e-8).unsqueeze(-1)
        hist = hist * query_f.unsqueeze(-1)

        hist_count = weights.sum(dim=-1)
        frac = (hist_count / float(window)).clamp(0.0, 1.0) * query_f
        return hist, frac

    def _pool_multi(
        self,
        event_emb: torch.Tensor,
        query_mask: torch.Tensor,
        history_mask: torch.Tensor,
        session_ids: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pooled = []
        for window in self.model_config.history_windows:
            pair_mask = self._make_pair_mask(
                history_mask=history_mask,
                window=window,
                session_ids=session_ids,
            )
            hist, frac = self._pool_from_pair_mask_mean(
                event_emb=event_emb,
                query_mask=query_mask,
                pair_mask=pair_mask,
                window=window,
            )
            pooled.append((hist, frac))
        return pooled

    def _pool_multi_future(
        self,
        event_emb: torch.Tensor,
        query_mask: torch.Tensor,
        future_visible_mask: torch.Tensor,
        event_ts_ns: torch.Tensor,
        session_ids: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pooled = []
        for window in self.model_config.future_windows:
            pair_mask = self._make_future_pair_mask(
                future_visible_mask=future_visible_mask,
                event_ts_ns=event_ts_ns,
                window=window,
                session_ids=session_ids,
            )
            hist, frac = self._pool_from_pair_mask_mean(
                event_emb=event_emb,
                query_mask=query_mask,
                pair_mask=pair_mask,
                window=window,
            )
            pooled.append((hist, frac))
        return pooled

    def _build_branch_features(
        self,
        reference_emb: torch.Tensor,
        pooled_list: List[Tuple[torch.Tensor, torch.Tensor]],
        include_current: bool,
        branch_weight: float = 1.0,
    ) -> List[torch.Tensor]:
        feats = []
        if include_current:
            feats.append(reference_emb)

        for hist, frac in pooled_list:
            hist_scaled = branch_weight * hist
            has_context = (frac > 0).float().unsqueeze(-1)

            hist_block = hist_scaled * has_context
            diff_block = (reference_emb - hist_scaled) * has_context
            prod_block = (reference_emb * hist_scaled) * has_context
            frac_block = frac.unsqueeze(-1) * has_context

            feats.extend([hist_block, diff_block, prod_block, frac_block])
        return feats

    def forward(
        self,
        cats: torch.Tensor,
        nums: torch.Tensor,
        valid_mask: torch.Tensor,
        session_ids: torch.Tensor,
        telemetry_hist_available: torch.Tensor,
        future_visible_mask: torch.Tensor,
        event_ts_ns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        stable_event_emb = self._encode_branch(
            cats=cats,
            nums=nums,
            valid_mask=valid_mask,
            cat_cols=self.model_config.stable_cat_cols,
            num_cols=self.model_config.stable_num_cols,
            encoder=self.stable_encoder,
        )

        telemetry_event_emb = self._encode_branch(
            cats=cats,
            nums=nums,
            valid_mask=valid_mask,
            cat_cols=self.model_config.telemetry_cat_cols,
            num_cols=self.model_config.telemetry_num_cols,
            encoder=self.telemetry_encoder,
        )

        telemetry_history_mask = valid_mask & telemetry_hist_available

        stable_hist_list = self._pool_multi(
            event_emb=stable_event_emb,
            query_mask=valid_mask,
            history_mask=valid_mask,
            session_ids=None,
        )

        telemetry_hist_list = self._pool_multi(
            event_emb=telemetry_event_emb,
            query_mask=valid_mask,
            history_mask=telemetry_history_mask,
            session_ids=None,
        )

        branch_features: Dict[str, torch.Tensor] = {}

        branch_features["stable_past"] = torch.cat(
            self._build_branch_features(
                reference_emb=stable_event_emb,
                pooled_list=stable_hist_list,
                include_current=True,
                branch_weight=1.0,
            ),
            dim=-1,
        )

        branch_features["telemetry_past"] = torch.cat(
            self._build_branch_features(
                reference_emb=telemetry_event_emb,
                pooled_list=telemetry_hist_list,
                include_current=True,
                branch_weight=1.0,
            ),
            dim=-1,
        )

        if self.model_config.use_session_branch:
            session_hist_list = self._pool_multi(
                event_emb=telemetry_event_emb,
                query_mask=valid_mask,
                history_mask=valid_mask,
                session_ids=session_ids,
            )
            branch_features["session"] = torch.cat(
                self._build_branch_features(
                    reference_emb=telemetry_event_emb,
                    pooled_list=session_hist_list,
                    include_current=False,
                    branch_weight=self.model_config.session_branch_weight,
                ),
                dim=-1,
            )

        if self.model_config.use_future_branches:
            stable_future_list = self._pool_multi_future(
                event_emb=stable_event_emb,
                query_mask=valid_mask,
                future_visible_mask=future_visible_mask,
                event_ts_ns=event_ts_ns,
                session_ids=None,
            )
            branch_features["future_stable"] = torch.cat(
                self._build_branch_features(
                    reference_emb=stable_event_emb,
                    pooled_list=stable_future_list,
                    include_current=False,
                    branch_weight=self.model_config.future_branch_weight,
                ),
                dim=-1,
            )

            telemetry_future_visible_mask = future_visible_mask & telemetry_hist_available
            telemetry_future_list = self._pool_multi_future(
                event_emb=telemetry_event_emb,
                query_mask=valid_mask,
                future_visible_mask=telemetry_future_visible_mask,
                event_ts_ns=event_ts_ns,
                session_ids=None,
            )
            branch_features["future_telemetry"] = torch.cat(
                self._build_branch_features(
                    reference_emb=telemetry_event_emb,
                    pooled_list=telemetry_future_list,
                    include_current=False,
                    branch_weight=self.model_config.future_branch_weight,
                ),
                dim=-1,
            )

        if self.model_config.label_history_num_cols:
            label_hist_block = self._select_num_block(nums, self.model_config.label_history_num_cols)
            label_hist_block = label_hist_block * valid_mask.unsqueeze(-1).float()
            branch_features["label_history"] = label_hist_block

        feats = [branch_features[name] for name in self.active_branch_names]

        fused = torch.cat(feats, dim=-1)
        fused = fused * valid_mask.unsqueeze(-1).float()

        red_logits = self.head_red(fused).squeeze(-1)
        suspicious_logits = self.head_suspicious(fused).squeeze(-1)
        ry_logits = self.head_ry(fused).squeeze(-1)

        p_red_main = torch.sigmoid(red_logits)
        p_suspicious = torch.sigmoid(suspicious_logits)
        p_red_given_suspicious = torch.sigmoid(ry_logits)

        p_red_aux = p_suspicious * p_red_given_suspicious
        final_score = (
            (1.0 - self.model_config.multitask_inference_blend) * p_red_main
            + self.model_config.multitask_inference_blend * p_red_aux
        )

        return {
            "red_logits": red_logits,
            "suspicious_logits": suspicious_logits,
            "ry_logits": ry_logits,
            "p_red_main": p_red_main,
            "p_suspicious": p_suspicious,
            "p_red_given_suspicious": p_red_given_suspicious,
            "p_red_aux": p_red_aux,
            "final_score": final_score,
        }


# ============================================================
# Train / predict
# ============================================================

def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_segments,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight_red: float,
    pos_weight_suspicious: float,
    pos_weight_ry: float,
    aux_loss_weight_suspicious: float,
    aux_loss_weight_red_yellow: float,
    grad_clip: float,
    device: torch.device,
) -> dict:
    model.train()

    criterion_red = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(pos_weight_red, dtype=torch.float32, device=device),
    )
    criterion_suspicious = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(pos_weight_suspicious, dtype=torch.float32, device=device),
    )
    criterion_ry = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(pos_weight_ry, dtype=torch.float32, device=device),
    )

    total_obj_sum = 0.0
    total_obj_count = 0
    red_sum = 0.0
    red_count = 0
    suspicious_sum = 0.0
    suspicious_count = 0
    ry_sum = 0.0
    ry_count = 0

    for batch in loader:
        cats = batch["cats"].to(device, non_blocking=True)
        nums = batch["nums"].to(device, non_blocking=True)
        labels_red = batch["labels"].to(device, non_blocking=True)
        target_raw = batch["target_raw"].to(device, non_blocking=True)
        session_ids = batch["session_ids"].to(device, non_blocking=True)
        telemetry_hist_available = batch["telemetry_hist_available"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        train_mask = batch["train_mask"].to(device, non_blocking=True)
        future_visible_mask = batch["future_visible_mask"].to(device, non_blocking=True)
        event_ts_ns = batch["event_ts_ns"].to(device, non_blocking=True)

        labeled_mask = train_mask & (target_raw > 0)
        ry_mask = train_mask & ((target_raw == 2) | (target_raw == 3))

        n_main = int(labeled_mask.sum().item())
        if n_main == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        out = model(
            cats,
            nums,
            valid_mask,
            session_ids,
            telemetry_hist_available,
            future_visible_mask,
            event_ts_ns,
        )

        losses = []

        loss_red_all = criterion_red(out["red_logits"], labels_red)
        loss_red = (loss_red_all * labeled_mask.float()).sum() / labeled_mask.float().sum().clamp_min(1.0)
        losses.append(loss_red)

        red_sum += float(loss_red.item()) * n_main
        red_count += n_main

        if aux_loss_weight_suspicious > 0:
            y_suspicious = ((target_raw == 2) | (target_raw == 3)).float()
            loss_susp_all = criterion_suspicious(out["suspicious_logits"], y_suspicious)
            loss_suspicious = (loss_susp_all * labeled_mask.float()).sum() / labeled_mask.float().sum().clamp_min(1.0)
            losses.append(aux_loss_weight_suspicious * loss_suspicious)

            suspicious_sum += float(loss_suspicious.item()) * n_main
            suspicious_count += n_main

        if aux_loss_weight_red_yellow > 0:
            y_ry = (target_raw == 3).float()
            n_ry = int(ry_mask.sum().item())
            if n_ry > 0:
                loss_ry_all = criterion_ry(out["ry_logits"], y_ry)
                loss_ry = (loss_ry_all * ry_mask.float()).sum() / ry_mask.float().sum().clamp_min(1.0)
                losses.append(aux_loss_weight_red_yellow * loss_ry)

                ry_sum += float(loss_ry.item()) * n_ry
                ry_count += n_ry

        loss = sum(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_obj_sum += float(loss.item()) * n_main
        total_obj_count += n_main

    return {
        "total_loss": (total_obj_sum / total_obj_count) if total_obj_count > 0 else float("nan"),
        "red_loss": (red_sum / red_count) if red_count > 0 else None,
        "suspicious_loss": (suspicious_sum / suspicious_count) if suspicious_count > 0 else None,
        "ry_loss": (ry_sum / ry_count) if ry_count > 0 else None,
    }


@torch.no_grad()
def predict_loader_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_pred_mask: bool = True,
) -> dict:
    model.eval()

    all_idx = []
    all_final = []
    all_red = []
    all_suspicious = []
    all_ry = []
    all_target_raw = []

    for batch in loader:
        cats = batch["cats"].to(device, non_blocking=True)
        nums = batch["nums"].to(device, non_blocking=True)
        session_ids = batch["session_ids"].to(device, non_blocking=True)
        telemetry_hist_available = batch["telemetry_hist_available"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        future_visible_mask = batch["future_visible_mask"].to(device, non_blocking=True)
        event_ts_ns = batch["event_ts_ns"].to(device, non_blocking=True)

        out = model(
            cats,
            nums,
            valid_mask,
            session_ids,
            telemetry_hist_available,
            future_visible_mask,
            event_ts_ns,
        )

        final_score = out["final_score"].cpu()
        red_score = out["p_red_main"].cpu()
        suspicious_score = out["p_suspicious"].cpu()
        ry_score = out["p_red_given_suspicious"].cpu()

        mask = batch["pred_mask"] if use_pred_mask else batch["train_mask"]
        row_idx = batch["row_idx"]
        target_raw = batch["target_raw"]

        all_idx.append(row_idx[mask].numpy())
        all_final.append(final_score[mask].numpy())
        all_red.append(red_score[mask].numpy())
        all_suspicious.append(suspicious_score[mask].numpy())
        all_ry.append(ry_score[mask].numpy())
        all_target_raw.append(target_raw[mask].numpy())

    if not all_idx:
        return {
            "idx": np.empty(0, dtype=np.int64),
            "final_score": np.empty(0, dtype=np.float32),
            "red_score": np.empty(0, dtype=np.float32),
            "suspicious_score": np.empty(0, dtype=np.float32),
            "ry_score": np.empty(0, dtype=np.float32),
            "target_raw": np.empty(0, dtype=np.int64),
        }

    return {
        "idx": np.concatenate(all_idx).astype(np.int64),
        "final_score": np.concatenate(all_final).astype(np.float32),
        "red_score": np.concatenate(all_red).astype(np.float32),
        "suspicious_score": np.concatenate(all_suspicious).astype(np.float32),
        "ry_score": np.concatenate(all_ry).astype(np.float32),
        "target_raw": np.concatenate(all_target_raw).astype(np.int64),
    }


def build_model(
    cat_cols: List[str],
    cat_cardinalities_total: Dict[str, int],
    cat_real_cardinalities: Dict[str, int],
    model_config: ModelConfig,
    device: torch.device,
) -> nn.Module:
    model = MultiTaskTwoBranchMultiWindowModel(
        all_cat_cols=cat_cols,
        cat_cardinalities_total=cat_cardinalities_total,
        cat_real_cardinalities=cat_real_cardinalities,
        model_config=model_config,
    )
    return model.to(device)


def save_model_checkpoint(
    path: Path,
    model: nn.Module,
    active_cat_cols: List[str],
    active_num_cols: List[str],
    model_config: ModelConfig,
    cat_cardinalities_total: Dict[str, int],
    cat_real_cardinalities: Dict[str, int],
    cat_mode_info: Dict[str, str],
    cat_encoding_stats: Dict[str, dict],
    cat_vocab_values: Dict[str, np.ndarray],
    extra_metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "cat_cardinalities_total": cat_cardinalities_total,
        "cat_real_cardinalities": cat_real_cardinalities,
        "cat_mode_info": cat_mode_info,
        "cat_encoding_stats": cat_encoding_stats,
        "cat_vocab_values": cat_vocab_values,
        "cat_cols": active_cat_cols,
        "num_cols": active_num_cols,
        "model_config": asdict(model_config),
        "preprocessor_policy": {
            "padding_id": 0,
            "unk_id": 1,
            "vocab_values_are_sorted_unique_train_visible_raw_values": True,
            "description": "raw_value -> code uses searchsorted over saved sorted vocab values; unseen values map to UNK=1.",
        },
    }
    payload.update(extra_metadata)
    torch.save(payload, path)


def run_training_job(
    job_name: str,
    store: EncodedSequenceStore,
    segments: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    train_future_visible_mask: np.ndarray,
    eval_future_visible_mask: np.ndarray,
    model_config: ModelConfig,
    cat_cardinalities_total: Dict[str, int],
    cat_real_cardinalities: Dict[str, int],
    cat_mode_info: Dict[str, str],
    cat_encoding_stats: Dict[str, dict],
    cat_vocab_values: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
    logger: logging.Logger,
    checkpoint_path: Path,
    checkpoint_extra_metadata: dict,
    es_mask: Optional[np.ndarray] = None,
    es_future_visible_mask: Optional[np.ndarray] = None,
    max_epochs: Optional[int] = None,
    use_early_stopping: bool = True,
) -> Tuple[nn.Module, dict, dict]:
    epochs = int(max_epochs if max_epochs is not None else args.max_epochs)

    train_segments = filter_segments_by_target_mask(segments, train_mask)
    eval_segments = filter_segments_by_target_mask(segments, eval_mask)

    train_dataset = SequenceSegmentDataset(
        store=store,
        segments=train_segments,
        row_train_mask=train_mask,
        row_pred_mask=np.zeros_like(train_mask, dtype=bool),
        row_future_visible_mask=train_future_visible_mask,
    )
    eval_dataset = SequenceSegmentDataset(
        store=store,
        segments=eval_segments,
        row_train_mask=np.zeros_like(eval_mask, dtype=bool),
        row_pred_mask=eval_mask,
        row_future_visible_mask=eval_future_visible_mask,
    )

    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, device=device)
    train_eval_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=False, device=device)
    eval_loader = make_loader(eval_dataset, batch_size=batch_size, shuffle=False, device=device)

    es_loader = None
    es_segments = np.empty((0, 4), dtype=np.int64)
    if es_mask is not None:
        es_segments = filter_segments_by_target_mask(segments, es_mask)
        if es_future_visible_mask is None:
            es_future_visible_mask = eval_future_visible_mask
        es_dataset = SequenceSegmentDataset(
            store=store,
            segments=es_segments,
            row_train_mask=np.zeros_like(es_mask, dtype=bool),
            row_pred_mask=es_mask,
            row_future_visible_mask=es_future_visible_mask,
        )
        es_loader = make_loader(es_dataset, batch_size=batch_size, shuffle=False, device=device)

    train_target_raw = store.target_raw[train_mask]
    red_binary = (train_target_raw == 3).astype(np.int64)
    suspicious_binary = np.isin(train_target_raw, [2, 3]).astype(np.int64)
    ry_subset = train_target_raw[np.isin(train_target_raw, [2, 3])]
    ry_binary = (ry_subset == 3).astype(np.int64)

    pos_weight_red = compute_pos_weight_from_binary(red_binary)
    pos_weight_suspicious = compute_pos_weight_from_binary(suspicious_binary)
    pos_weight_ry = compute_pos_weight_from_binary(ry_binary) if len(ry_binary) > 0 else 1.0

    model = build_model(
        cat_cols=store.active_cat_cols,
        cat_cardinalities_total=cat_cardinalities_total,
        cat_real_cardinalities=cat_real_cardinalities,
        model_config=model_config,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    logger.info(
        "%s | train_rows=%d eval_rows=%d es_rows=%d | train_red=%d train_yellow=%d train_suspicious=%d | epochs=%d early_stopping=%s",
        job_name,
        int(train_mask.sum()),
        int(eval_mask.sum()),
        int(es_mask.sum()) if es_mask is not None else 0,
        int(np.sum(store.target_raw[train_mask] == 3)),
        int(np.sum(store.target_raw[train_mask] == 2)),
        int(np.sum(np.isin(store.target_raw[train_mask], [2, 3]))),
        epochs,
        use_early_stopping and (es_loader is not None),
    )

    best_state = None
    best_monitor = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    epoch_history = []

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            pos_weight_red=pos_weight_red,
            pos_weight_suspicious=pos_weight_suspicious,
            pos_weight_ry=pos_weight_ry,
            aux_loss_weight_suspicious=args.aux_loss_weight_suspicious,
            aux_loss_weight_red_yellow=args.aux_loss_weight_red_yellow,
            grad_clip=args.grad_clip,
            device=device,
        )

        train_final_ap = None
        train_red_ap = None
        train_suspicious_ap = None
        train_ry_ap = None

        if epoch % args.train_ap_every_n_epochs == 0:
            train_pred_epoch = predict_loader_all(
                model=model,
                loader=train_eval_loader,
                device=device,
                use_pred_mask=False,
            )
            train_metrics_epoch = compute_ap_metrics(
                target_raw=train_pred_epoch["target_raw"],
                final_score=train_pred_epoch["final_score"],
                red_score=train_pred_epoch["red_score"],
                suspicious_score=train_pred_epoch["suspicious_score"],
                ry_score=train_pred_epoch["ry_score"],
            )
            train_final_ap = train_metrics_epoch["final_ap"]
            train_red_ap = train_metrics_epoch["red_ap"]
            train_suspicious_ap = train_metrics_epoch["suspicious_ap"]
            train_ry_ap = train_metrics_epoch["ry_ap"]

        es_metrics_epoch = None
        if es_loader is not None:
            es_pred_epoch = predict_loader_all(
                model=model,
                loader=es_loader,
                device=device,
                use_pred_mask=True,
            )
            es_metrics_epoch = compute_ap_metrics(
                target_raw=es_pred_epoch["target_raw"],
                final_score=es_pred_epoch["final_score"],
                red_score=es_pred_epoch["red_score"],
                suspicious_score=es_pred_epoch["suspicious_score"],
                ry_score=es_pred_epoch["ry_score"],
            )

        epoch_rec = {
            "epoch": epoch,
            "train_total_loss": float(train_stats["total_loss"]) if not np.isnan(train_stats["total_loss"]) else None,
            "train_red_loss": train_stats["red_loss"],
            "train_suspicious_loss": train_stats["suspicious_loss"],
            "train_ry_loss": train_stats["ry_loss"],
            "train_final_ap": train_final_ap,
            "train_red_ap": train_red_ap,
            "train_suspicious_ap": train_suspicious_ap,
            "train_ry_ap": train_ry_ap,
            "es_final_ap": None if es_metrics_epoch is None else es_metrics_epoch["final_ap"],
            "es_red_ap": None if es_metrics_epoch is None else es_metrics_epoch["red_ap"],
            "es_suspicious_ap": None if es_metrics_epoch is None else es_metrics_epoch["suspicious_ap"],
            "es_ry_ap": None if es_metrics_epoch is None else es_metrics_epoch["ry_ap"],
        }
        epoch_history.append(epoch_rec)

        logger.info(
            "%s | epoch %d | total_loss=%s | red_loss=%s | suspicious_loss=%s | ry_loss=%s | train_final_ap=%s | train_red_ap=%s | train_suspicious_ap=%s | train_ry_ap=%s | es_final_ap=%s | es_red_ap=%s | es_suspicious_ap=%s | es_ry_ap=%s",
            job_name,
            epoch,
            fmt_optional(epoch_rec["train_total_loss"]),
            fmt_optional(epoch_rec["train_red_loss"]),
            fmt_optional(epoch_rec["train_suspicious_loss"]),
            fmt_optional(epoch_rec["train_ry_loss"]),
            "skip" if train_final_ap is None else f"{train_final_ap:.6f}",
            "skip" if train_red_ap is None else f"{train_red_ap:.6f}",
            "skip" if train_suspicious_ap is None else f"{train_suspicious_ap:.6f}",
            "skip" if train_ry_ap is None else f"{train_ry_ap:.6f}",
            fmt_optional(epoch_rec["es_final_ap"]),
            fmt_optional(epoch_rec["es_red_ap"]),
            fmt_optional(epoch_rec["es_suspicious_ap"]),
            fmt_optional(epoch_rec["es_ry_ap"]),
        )

        if use_early_stopping and es_metrics_epoch is not None:
            monitor = -1.0 if es_metrics_epoch["final_ap"] is None else float(es_metrics_epoch["final_ap"])
            if monitor > best_monitor:
                best_monitor = monitor
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                logger.info("%s | early stopping at epoch %d by final AP", job_name, epoch)
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = max(1, len(epoch_history))
        best_monitor = float("nan")

    model.load_state_dict(best_state)

    train_pred_best = predict_loader_all(
        model=model,
        loader=train_eval_loader,
        device=device,
        use_pred_mask=False,
    )
    train_metrics_best = compute_ap_metrics(
        target_raw=train_pred_best["target_raw"],
        final_score=train_pred_best["final_score"],
        red_score=train_pred_best["red_score"],
        suspicious_score=train_pred_best["suspicious_score"],
        ry_score=train_pred_best["ry_score"],
    )

    eval_pred_best = predict_loader_all(
        model=model,
        loader=eval_loader,
        device=device,
        use_pred_mask=True,
    )
    eval_metrics_best = compute_ap_metrics(
        target_raw=eval_pred_best["target_raw"],
        final_score=eval_pred_best["final_score"],
        red_score=eval_pred_best["red_score"],
        suspicious_score=eval_pred_best["suspicious_score"],
        ry_score=eval_pred_best["ry_score"],
    )

    es_metrics_best = None
    if es_loader is not None:
        es_pred_best = predict_loader_all(
            model=model,
            loader=es_loader,
            device=device,
            use_pred_mask=True,
        )
        es_metrics_best = compute_ap_metrics(
            target_raw=es_pred_best["target_raw"],
            final_score=es_pred_best["final_score"],
            red_score=es_pred_best["red_score"],
            suspicious_score=es_pred_best["suspicious_score"],
            ry_score=es_pred_best["ry_score"],
        )

    logger.info(
        "%s | best_epoch=%d | train_final_ap=%s | train_red_ap=%s | train_suspicious_ap=%s | train_ry_ap=%s | eval_final_ap=%s | eval_red_ap=%s | eval_suspicious_ap=%s | eval_ry_ap=%s | es_final_ap=%s | es_red_ap=%s | es_suspicious_ap=%s | es_ry_ap=%s",
        job_name,
        best_epoch,
        fmt_optional(train_metrics_best["final_ap"]),
        fmt_optional(train_metrics_best["red_ap"]),
        fmt_optional(train_metrics_best["suspicious_ap"]),
        fmt_optional(train_metrics_best["ry_ap"]),
        fmt_optional(eval_metrics_best["final_ap"]),
        fmt_optional(eval_metrics_best["red_ap"]),
        fmt_optional(eval_metrics_best["suspicious_ap"]),
        fmt_optional(eval_metrics_best["ry_ap"]),
        fmt_optional(None if es_metrics_best is None else es_metrics_best["final_ap"]),
        fmt_optional(None if es_metrics_best is None else es_metrics_best["red_ap"]),
        fmt_optional(None if es_metrics_best is None else es_metrics_best["suspicious_ap"]),
        fmt_optional(None if es_metrics_best is None else es_metrics_best["ry_ap"]),
    )

    save_model_checkpoint(
        path=checkpoint_path,
        model=model,
        active_cat_cols=store.active_cat_cols,
        active_num_cols=store.active_num_cols,
        model_config=model_config,
        cat_cardinalities_total=cat_cardinalities_total,
        cat_real_cardinalities=cat_real_cardinalities,
        cat_mode_info=cat_mode_info,
        cat_encoding_stats=cat_encoding_stats,
        cat_vocab_values=cat_vocab_values,
        extra_metadata={
            **checkpoint_extra_metadata,
            "training_summary": {
                "best_epoch": int(best_epoch),
                "best_monitor_value": None if np.isnan(best_monitor) else float(best_monitor),
                "train_metrics": train_metrics_best,
                "eval_metrics": eval_metrics_best,
                "es_metrics": es_metrics_best,
                "epoch_history": epoch_history,
                "pos_weight_red": float(pos_weight_red),
                "pos_weight_suspicious": float(pos_weight_suspicious),
                "pos_weight_ry": float(pos_weight_ry),
            },
        },
    )

    job_info = {
        "best_epoch": int(best_epoch),
        "best_monitor_value": None if np.isnan(best_monitor) else float(best_monitor),
        "train_metrics": train_metrics_best,
        "eval_metrics": eval_metrics_best,
        "es_metrics": es_metrics_best,
        "epoch_history": epoch_history,
        "model_path": str(checkpoint_path),
        "pos_weight_red": float(pos_weight_red),
        "pos_weight_suspicious": float(pos_weight_suspicious),
        "pos_weight_ry": float(pos_weight_ry),
        "train_segments": int(len(train_segments)),
        "eval_segments": int(len(eval_segments)),
        "es_segments": int(len(es_segments)) if es_mask is not None else 0,
    }

    return model, job_info, eval_pred_best


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    logger = setup_logging(output_dir)

    set_seed(args.random_state)
    torch.set_num_threads(args.threads)

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    batch_size = args.batch_size_gpu if device.type == "cuda" else args.batch_size_cpu
    active_cat_cols = get_active_cat_cols()
    active_num_cols = get_active_num_cols(use_label_history=args.use_label_history)
    model_config = build_model_config(args, active_cat_cols, active_num_cols)

    save_json(output_dir / "run_options.json", vars(args))

    logger.info("Device: %s", device)
    logger.info("Threads: %d", args.threads)
    logger.info("Random state: %d", args.random_state)
    logger.info("Output dir: %s", output_dir)
    logger.info("History pooling: mean")
    logger.info("History windows: %s", args.history_windows)
    logger.info("Future windows: %s", args.future_windows)
    logger.info("Use future branches: %s", args.use_future_branches)
    logger.info("Future max hours: %.4f", args.future_max_hours)
    logger.info("Future branch weight: %.4f", args.future_branch_weight)
    logger.info("Use label history: %s", args.use_label_history)
    logger.info(
        "Multitask config | aux_suspicious=%.4f aux_ry=%.4f inference_blend=%.4f",
        args.aux_loss_weight_suspicious,
        args.aux_loss_weight_red_yellow,
        args.multitask_inference_blend,
    )
    logger.info(
        "Session config | use_session_branch=%s session_branch_weight=%.4f",
        args.use_session_branch,
        args.session_branch_weight,
    )
    logger.info("Model config: %s", asdict(model_config))

    store, global_label_history_stats = load_and_preprocess(
        input_path=args.input_path,
        active_cat_cols=active_cat_cols,
        use_label_history=args.use_label_history,
        logger=logger,
    )

    if store.active_num_cols != active_num_cols:
        raise ValueError("Loaded numerical columns do not match expected active_num_cols.")

    logger.info(
        "Building segments with last_n=%d, max_pred_seq_len=%d ...",
        args.last_n,
        args.max_pred_seq_len,
    )
    segments = build_segments(
        customer_id=store.customer_id,
        event_dttm=store.event_dttm,
        last_n=args.last_n,
        max_pred_seq_len=args.max_pred_seq_len,
        use_future_branches=args.use_future_branches,
        future_windows=args.future_windows,
        future_max_hours=args.future_max_hours,
    )
    logger.info("Built %d segments", len(segments))

    folds = make_monthly_folds(
        cv_mode=args.cv_mode,
        sliding_train_months=args.sliding_train_months,
    )
    logger.info("CV mode=%s | folds=%d", args.cv_mode, len(folds))
    logger.info("Fold IDs: %s", [f["fold_idx"] for f in folds])

    train_global_index = np.flatnonzero(store.dataset_id == TRAIN_DATASET_ID)
    test_global_index = np.flatnonzero(store.dataset_id == TEST_DATASET_ID)

    train_event_id = store.event_id[train_global_index]
    train_customer_id = store.customer_id[train_global_index]
    train_event_dttm = store.event_dttm[train_global_index]
    train_target = store.target_raw[train_global_index]
    test_event_id = store.event_id[test_global_index]

    oof_pred_final = np.full(len(train_global_index), np.nan, dtype=np.float32)
    oof_pred_red = np.full(len(train_global_index), np.nan, dtype=np.float32)
    oof_pred_suspicious = np.full(len(train_global_index), np.nan, dtype=np.float32)
    oof_pred_ry = np.full(len(train_global_index), np.nan, dtype=np.float32)
    oof_fold = np.full(len(train_global_index), -1, dtype=np.int16)

    test_fold_pred_arrays: List[np.ndarray] = []
    test_fold_ids: List[int] = []
    fold_metrics_list = []
    best_epochs = []

    reference_cat_info = {
        "modes": {},
        "real_cardinalities": {},
        "total_cardinalities_with_padding": {},
        "encoding_stats": {},
    }

    for fold in folds:
        fold_idx = fold["fold_idx"]

        fold_vocab_mask = store.event_dttm < dt64(fold["valid_start"])
        (
            fold_encoded_store,
            fold_cat_total,
            fold_cat_real,
            fold_cat_mode_info,
            fold_cat_encoding_stats,
            fold_cat_vocab_values,
        ) = build_encoded_store(
            store=store,
            vocab_fit_mask=fold_vocab_mask,
            one_hot_max_cardinality=model_config.one_hot_max_cardinality,
            logger=logger,
            tag=f"Fold {fold_idx}",
        )

        reference_cat_info["modes"] = fold_cat_mode_info
        reference_cat_info["real_cardinalities"] = fold_cat_real
        reference_cat_info["total_cardinalities_with_padding"] = fold_cat_total
        reference_cat_info["encoding_stats"] = fold_cat_encoding_stats

        train_base_mask = (
            (fold_encoded_store.dataset_id == TRAIN_DATASET_ID)
            & (fold_encoded_store.event_dttm >= dt64(fold["train_start"]))
            & (fold_encoded_store.event_dttm < dt64(fold["train_end_exclusive"]))
        )
        valid_full_mask = (
            (fold_encoded_store.dataset_id == TRAIN_DATASET_ID)
            & (fold_encoded_store.event_dttm >= dt64(fold["valid_start"]))
            & (fold_encoded_store.event_dttm < dt64(fold["valid_end_exclusive"]))
        )

        rng_train = np.random.default_rng(args.random_state + 1000 * fold_idx + 11)
        rng_es = np.random.default_rng(args.random_state + 1000 * fold_idx + 29)

        train_fit_mask = sample_target1_mask(
            base_mask=train_base_mask,
            target_raw=fold_encoded_store.target_raw,
            frac=args.target1_sample_frac,
            rng=rng_train,
        )
        valid_es_mask = sample_target1_mask(
            base_mask=valid_full_mask,
            target_raw=fold_encoded_store.target_raw,
            frac=args.target1_sample_frac,
            rng=rng_es,
        )

        model, job_info, valid_pred = run_training_job(
            job_name=f"Fold {fold_idx}",
            store=fold_encoded_store,
            segments=segments,
            train_mask=train_fit_mask,
            eval_mask=valid_full_mask,
            train_future_visible_mask=train_base_mask,
            eval_future_visible_mask=valid_full_mask,
            model_config=model_config,
            cat_cardinalities_total=fold_cat_total,
            cat_real_cardinalities=fold_cat_real,
            cat_mode_info=fold_cat_mode_info,
            cat_encoding_stats=fold_cat_encoding_stats,
            cat_vocab_values=fold_cat_vocab_values,
            args=args,
            device=device,
            batch_size=batch_size,
            logger=logger,
            checkpoint_path=output_dir / "models" / f"fold_{fold_idx:02d}.pt",
            checkpoint_extra_metadata={
                "fold": fold,
                "args": vars(args),
                "label_history_policy": {
                    "enabled": bool(args.use_label_history),
                    "lag_days": 0,
                    "hidden_intervals": [],
                },
            },
            es_mask=valid_es_mask,
            es_future_visible_mask=valid_full_mask,
            max_epochs=args.max_epochs,
            use_early_stopping=True,
        )

        best_epochs.append(int(job_info["best_epoch"]))

        valid_pos = map_positions(train_global_index, valid_pred["idx"])
        oof_pred_final[valid_pos] = valid_pred["final_score"]
        oof_pred_red[valid_pos] = valid_pred["red_score"]
        oof_pred_suspicious[valid_pos] = valid_pred["suspicious_score"]
        oof_pred_ry[valid_pos] = valid_pred["ry_score"]
        oof_fold[valid_pos] = fold_idx

        fold_valid_df = pl.DataFrame(
            {
                "event_id": fold_encoded_store.event_id[valid_pred["idx"]],
                "customer_id": fold_encoded_store.customer_id[valid_pred["idx"]],
                "event_dttm": fold_encoded_store.event_dttm[valid_pred["idx"]],
                "target": fold_encoded_store.target_raw[valid_pred["idx"]],
                "predict": valid_pred["final_score"],
                "predict_final": valid_pred["final_score"],
                "predict_red_main": valid_pred["red_score"],
                "predict_suspicious": valid_pred["suspicious_score"],
                "predict_red_given_suspicious": valid_pred["ry_score"],
                "fold_idx": np.full(len(valid_pred["idx"]), fold_idx, dtype=np.int16),
            }
        )
        fold_valid_df.write_parquet(output_dir / f"valid_predictions_fold_{fold_idx:02d}.parquet")

        test_mask = fold_encoded_store.dataset_id == TEST_DATASET_ID
        test_segments = filter_segments_by_target_mask(segments, test_mask)
        test_dataset = SequenceSegmentDataset(
            store=fold_encoded_store,
            segments=test_segments,
            row_train_mask=np.zeros_like(test_mask, dtype=bool),
            row_pred_mask=test_mask,
            row_future_visible_mask=test_mask,
        )
        test_loader = make_loader(test_dataset, batch_size=batch_size, shuffle=False, device=device)

        test_pred_all = predict_loader_all(
            model=model,
            loader=test_loader,
            device=device,
            use_pred_mask=True,
        )

        fold_pred_full = np.full(len(test_global_index), np.nan, dtype=np.float32)
        test_pos = map_positions(test_global_index, test_pred_all["idx"])
        fold_pred_full[test_pos] = test_pred_all["final_score"]

        test_fold_pred_arrays.append(fold_pred_full)
        test_fold_ids.append(fold_idx)

        fold_test_df = pl.DataFrame(
            {
                "event_id": test_event_id,
                "predict": fold_pred_full,
                "fold_idx": np.full(len(test_event_id), fold_idx, dtype=np.int16),
            }
        )
        fold_test_df.write_parquet(output_dir / f"test_predictions_fold_{fold_idx:02d}.parquet")

        fold_metrics = {
            "fold_idx": fold_idx,
            "train_start": fold["train_start"].strftime("%Y-%m-%d"),
            "train_end_exclusive": fold["train_end_exclusive"].strftime("%Y-%m-%d"),
            "valid_start": fold["valid_start"].strftime("%Y-%m-%d"),
            "valid_end_exclusive": fold["valid_end_exclusive"].strftime("%Y-%m-%d"),

            "use_label_history": bool(args.use_label_history),
            "label_history_lag_days": 0,

            "use_future_branches": bool(args.use_future_branches),
            "future_max_hours": float(args.future_max_hours),
            "future_branch_weight": float(args.future_branch_weight),
            "future_windows": list(args.future_windows),

            "train_base_rows": int(train_base_mask.sum()),
            "train_fit_rows": int(train_fit_mask.sum()),
            "train_base_red": int(np.sum(fold_encoded_store.target_raw[train_base_mask] == 3)),
            "train_fit_red": int(np.sum(fold_encoded_store.target_raw[train_fit_mask] == 3)),
            "train_fit_yellow": int(np.sum(fold_encoded_store.target_raw[train_fit_mask] == 2)),
            "train_fit_suspicious": int(np.sum(np.isin(fold_encoded_store.target_raw[train_fit_mask], [2, 3]))),

            "valid_full_rows": int(valid_full_mask.sum()),
            "valid_full_red": int(np.sum(fold_encoded_store.target_raw[valid_full_mask] == 3)),
            "valid_full_yellow": int(np.sum(fold_encoded_store.target_raw[valid_full_mask] == 2)),
            "valid_full_suspicious": int(np.sum(np.isin(fold_encoded_store.target_raw[valid_full_mask], [2, 3]))),

            "valid_es_rows": int(valid_es_mask.sum()),
            "valid_es_red": int(np.sum(fold_encoded_store.target_raw[valid_es_mask] == 3)),
            "valid_es_yellow": int(np.sum(fold_encoded_store.target_raw[valid_es_mask] == 2)),
            "valid_es_suspicious": int(np.sum(np.isin(fold_encoded_store.target_raw[valid_es_mask], [2, 3]))),

            "train_segments": job_info["train_segments"],
            "valid_es_segments": job_info["es_segments"],
            "valid_full_segments": job_info["eval_segments"],

            "pos_weight_red": job_info["pos_weight_red"],
            "pos_weight_suspicious": job_info["pos_weight_suspicious"],
            "pos_weight_ry": job_info["pos_weight_ry"],

            "best_epoch": job_info["best_epoch"],
            "best_es_final_ap_during_training": job_info["best_monitor_value"],

            "recalculated_train_final_ap": job_info["train_metrics"]["final_ap"],
            "recalculated_train_red_ap": job_info["train_metrics"]["red_ap"],
            "recalculated_train_suspicious_ap": job_info["train_metrics"]["suspicious_ap"],
            "recalculated_train_ry_ap": job_info["train_metrics"]["ry_ap"],

            "recalculated_es_final_ap": None if job_info["es_metrics"] is None else job_info["es_metrics"]["final_ap"],
            "recalculated_es_red_ap": None if job_info["es_metrics"] is None else job_info["es_metrics"]["red_ap"],
            "recalculated_es_suspicious_ap": None if job_info["es_metrics"] is None else job_info["es_metrics"]["suspicious_ap"],
            "recalculated_es_ry_ap": None if job_info["es_metrics"] is None else job_info["es_metrics"]["ry_ap"],

            "recalculated_full_valid_final_ap": job_info["eval_metrics"]["final_ap"],
            "recalculated_full_valid_red_ap": job_info["eval_metrics"]["red_ap"],
            "recalculated_full_valid_suspicious_ap": job_info["eval_metrics"]["suspicious_ap"],
            "recalculated_full_valid_ry_ap": job_info["eval_metrics"]["ry_ap"],

            "cat_mode_info": fold_cat_mode_info,
            "cat_real_cardinalities": fold_cat_real,
            "cat_total_cardinalities_with_padding": fold_cat_total,
            "cat_encoding_stats": fold_cat_encoding_stats,
            "cat_vocab_sizes": {c: int(len(v)) for c, v in fold_cat_vocab_values.items()},
            "epoch_history": job_info["epoch_history"],
            "model_path": job_info["model_path"],
        }
        fold_metrics_list.append(fold_metrics)

        del model
        del fold_encoded_store
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_available_mask = ~np.isnan(oof_pred_final)
    oof_metrics = compute_ap_metrics(
        target_raw=train_target[oof_available_mask],
        final_score=oof_pred_final[oof_available_mask],
        red_score=oof_pred_red[oof_available_mask],
        suspicious_score=oof_pred_suspicious[oof_available_mask],
        ry_score=oof_pred_ry[oof_available_mask],
    ) if np.any(oof_available_mask) else {
        "final_ap": float("nan"),
        "red_ap": float("nan"),
        "suspicious_ap": float("nan"),
        "ry_ap": float("nan"),
    }

    oof_df = pl.DataFrame(
        {
            "event_id": train_event_id,
            "customer_id": train_customer_id,
            "event_dttm": train_event_dttm,
            "target": train_target,
            "predict": oof_pred_final,
            "predict_final": oof_pred_final,
            "predict_red_main": oof_pred_red,
            "predict_suspicious": oof_pred_suspicious,
            "predict_red_given_suspicious": oof_pred_ry,
            "fold_idx": oof_fold,
            "oof_available": oof_available_mask,
        }
    )
    oof_df.write_parquet(output_dir / "oof_predictions.parquet")

    mean_fold_final_ap = mean_ignore_none(
        [fm.get("recalculated_full_valid_final_ap") for fm in fold_metrics_list]
    )
    mean_fold_red_ap = mean_ignore_none(
        [fm.get("recalculated_full_valid_red_ap") for fm in fold_metrics_list]
    )
    mean_fold_suspicious_ap = mean_ignore_none(
        [fm.get("recalculated_full_valid_suspicious_ap") for fm in fold_metrics_list]
    )
    mean_fold_ry_ap = mean_ignore_none(
        [fm.get("recalculated_full_valid_ry_ap") for fm in fold_metrics_list]
    )

    mean_fold_es_final_ap = mean_ignore_none(
        [fm.get("recalculated_es_final_ap") for fm in fold_metrics_list]
    )
    mean_fold_es_red_ap = mean_ignore_none(
        [fm.get("recalculated_es_red_ap") for fm in fold_metrics_list]
    )
    mean_fold_es_suspicious_ap = mean_ignore_none(
        [fm.get("recalculated_es_suspicious_ap") for fm in fold_metrics_list]
    )
    mean_fold_es_ry_ap = mean_ignore_none(
        [fm.get("recalculated_es_ry_ap") for fm in fold_metrics_list]
    )

    cv_ensemble_pred = build_ensemble_prediction(test_fold_pred_arrays)
    cv_ensemble_pq = output_dir / "test_predictions_cv_ensemble.parquet"
    cv_ensemble_csv = output_dir / "submission_cv_ensemble.csv"
    save_event_prediction_files(
        event_id=test_event_id,
        predict=cv_ensemble_pred,
        parquet_path=cv_ensemble_pq,
        csv_path=cv_ensemble_csv,
    )

    ensemble_variant_info = {
        "cv_ensemble": {
            "fold_ids": test_fold_ids,
            "n_folds": len(test_fold_ids),
            "parquet": str(cv_ensemble_pq),
            "csv": str(cv_ensemble_csv),
        }
    }

    final_model_trained = False
    final_model_test_path: Optional[Path] = None
    final_model_submission_csv: Optional[Path] = None
    final_model_epochs: Optional[int] = None

    if args.train_final_model:
        final_model_epochs = int(np.median(best_epochs)) if best_epochs else args.max_epochs
        final_model_epochs = max(1, final_model_epochs)

        final_vocab_mask = store.event_dttm < dt64(TRAIN_END_EXCLUSIVE)
        (
            final_encoded_store,
            final_cat_total,
            final_cat_real,
            final_cat_mode_info,
            final_cat_encoding_stats,
            final_cat_vocab_values,
        ) = build_encoded_store(
            store=store,
            vocab_fit_mask=final_vocab_mask,
            one_hot_max_cardinality=model_config.one_hot_max_cardinality,
            logger=logger,
            tag="Final model",
        )

        reference_cat_info["modes"] = final_cat_mode_info
        reference_cat_info["real_cardinalities"] = final_cat_real
        reference_cat_info["total_cardinalities_with_padding"] = final_cat_total
        reference_cat_info["encoding_stats"] = final_cat_encoding_stats

        train_base_mask = final_encoded_store.dataset_id == TRAIN_DATASET_ID
        rng_final = np.random.default_rng(args.random_state + 999_999)
        train_fit_mask = sample_target1_mask(
            base_mask=train_base_mask,
            target_raw=final_encoded_store.target_raw,
            frac=args.target1_sample_frac,
            rng=rng_final,
        )

        final_model, final_job_info, _ = run_training_job(
            job_name="Final model",
            store=final_encoded_store,
            segments=segments,
            train_mask=train_fit_mask,
            eval_mask=train_fit_mask,
            train_future_visible_mask=train_base_mask,
            eval_future_visible_mask=train_base_mask,
            model_config=model_config,
            cat_cardinalities_total=final_cat_total,
            cat_real_cardinalities=final_cat_real,
            cat_mode_info=final_cat_mode_info,
            cat_encoding_stats=final_cat_encoding_stats,
            cat_vocab_values=final_cat_vocab_values,
            args=args,
            device=device,
            batch_size=batch_size,
            logger=logger,
            checkpoint_path=output_dir / "models" / "final_model.pt",
            checkpoint_extra_metadata={
                "args": vars(args),
                "final_model_epochs": final_model_epochs,
                "label_history_policy": {
                    "enabled": bool(args.use_label_history),
                    "lag_days": 0,
                    "hidden_intervals": [],
                },
            },
            es_mask=None,
            es_future_visible_mask=None,
            max_epochs=final_model_epochs,
            use_early_stopping=False,
        )

        final_test_mask = final_encoded_store.dataset_id == TEST_DATASET_ID
        final_test_segments = filter_segments_by_target_mask(segments, final_test_mask)
        final_test_dataset = SequenceSegmentDataset(
            store=final_encoded_store,
            segments=final_test_segments,
            row_train_mask=np.zeros_like(final_test_mask, dtype=bool),
            row_pred_mask=final_test_mask,
            row_future_visible_mask=final_test_mask,
        )
        final_test_loader = make_loader(final_test_dataset, batch_size=batch_size, shuffle=False, device=device)

        final_test_pred_all = predict_loader_all(
            model=final_model,
            loader=final_test_loader,
            device=device,
            use_pred_mask=True,
        )

        final_test_pos = map_positions(test_global_index, final_test_pred_all["idx"])
        final_test_pred_full = np.full(len(test_global_index), np.nan, dtype=np.float32)
        final_test_pred_full[final_test_pos] = final_test_pred_all["final_score"]

        final_model_test_path = output_dir / "test_predictions_final_model.parquet"
        final_model_submission_csv = output_dir / "submission_final_model.csv"

        save_event_prediction_files(
            event_id=test_event_id,
            predict=final_test_pred_full,
            parquet_path=final_model_test_path,
            csv_path=final_model_submission_csv,
        )

        save_json(
            output_dir / "final_model_training.json",
            {
                "epochs": final_model_epochs,
                "training_summary": final_job_info,
                "model_config": asdict(model_config),
                "cat_mode_info": final_cat_mode_info,
                "cat_real_cardinalities": final_cat_real,
                "cat_total_cardinalities_with_padding": final_cat_total,
                "cat_encoding_stats": final_cat_encoding_stats,
                "cat_vocab_sizes": {c: int(len(v)) for c, v in final_cat_vocab_values.items()},
                "global_label_history_stats": global_label_history_stats,
                "model_path": str(output_dir / "models" / "final_model.pt"),
            },
        )

        final_model_trained = True

        del final_model
        del final_encoded_store
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_json(output_dir / "fold_metrics_detailed.json", {"folds": fold_metrics_list})

    fold_summary_keys = [
        "fold_idx",
        "best_epoch",
        "best_es_final_ap_during_training",
        "recalculated_train_final_ap",
        "recalculated_train_red_ap",
        "recalculated_train_suspicious_ap",
        "recalculated_train_ry_ap",
        "recalculated_es_final_ap",
        "recalculated_es_red_ap",
        "recalculated_es_suspicious_ap",
        "recalculated_es_ry_ap",
        "recalculated_full_valid_final_ap",
        "recalculated_full_valid_red_ap",
        "recalculated_full_valid_suspicious_ap",
        "recalculated_full_valid_ry_ap",
        "train_fit_rows",
        "valid_full_rows",
        "use_label_history",
        "use_future_branches",
        "future_max_hours",
        "future_windows",
    ]

    results = {
        "status": "ok",
        "device": str(device),
        "cv_mode": args.cv_mode,
        "n_folds": len(folds),

        "oof_final_ap_scored_rows": oof_metrics["final_ap"],
        "oof_red_ap_scored_rows": oof_metrics["red_ap"],
        "oof_suspicious_ap_scored_rows": oof_metrics["suspicious_ap"],
        "oof_ry_ap_scored_rows": oof_metrics["ry_ap"],

        "mean_fold_final_ap": mean_fold_final_ap,
        "mean_fold_red_ap": mean_fold_red_ap,
        "mean_fold_suspicious_ap": mean_fold_suspicious_ap,
        "mean_fold_ry_ap": mean_fold_ry_ap,

        "mean_fold_es_final_ap": mean_fold_es_final_ap,
        "mean_fold_es_red_ap": mean_fold_es_red_ap,
        "mean_fold_es_suspicious_ap": mean_fold_es_suspicious_ap,
        "mean_fold_es_ry_ap": mean_fold_es_ry_ap,

        "oof_rows_total": int(len(train_global_index)),
        "oof_rows_scored": int(oof_available_mask.sum()),
        "oof_coverage_ratio": float(oof_available_mask.mean()) if len(oof_available_mask) else 0.0,

        "test_rows": int(len(test_global_index)),

        "train_final_model": bool(args.train_final_model),
        "final_model_trained": final_model_trained,
        "final_model_epochs": final_model_epochs,
        "use_label_history": bool(args.use_label_history),

        "sampling": {
            "target1_sample_frac": float(args.target1_sample_frac),
            "description": "Applied to dataset=1,target=1 rows in fold train, ES subset and final-model training.",
        },

        "monitoring": {
            "train_ap_every_n_epochs": int(args.train_ap_every_n_epochs),
            "description": "Train APs are computed every N epochs; epoch logs also include ES final/red/suspicious/RY AP.",
            "early_stopping_metric": "final_ap",
        },

        "pooling": {
            "history_pooling": "mean",
            "history_windows": list(args.history_windows),
            "future_windows": list(args.future_windows),
        },

        "multitask": {
            "aux_loss_weight_suspicious": float(args.aux_loss_weight_suspicious),
            "aux_loss_weight_red_yellow": float(args.aux_loss_weight_red_yellow),
            "multitask_inference_blend": float(args.multitask_inference_blend),
        },

        "session": {
            "use_session_branch": bool(args.use_session_branch),
            "session_branch_weight": float(args.session_branch_weight),
        },

        "future_branches": {
            "enabled": bool(args.use_future_branches),
            "future_max_hours": float(args.future_max_hours),
            "future_branch_weight": float(args.future_branch_weight),
            "future_windows": list(args.future_windows),
            "description": "Optional future raw-event branches over later events within a limited horizon.",
        },

        "label_history_policy": {
            "enabled": bool(args.use_label_history),
            "lag_days": 0,
            "hidden_intervals": [],
            "global_stats": global_label_history_stats,
            "description": "No-lag causal label-history built on all past visible train labels.",
        },

        "categorical_encoding": {
            "active_cat_cols": active_cat_cols,
            "active_num_cols": active_num_cols,
            "one_hot_max_cardinality": int(args.one_hot_max_cardinality),
            "reference_cat_modes": reference_cat_info["modes"],
            "reference_cat_real_cardinalities": reference_cat_info["real_cardinalities"],
            "reference_cat_total_cardinalities_with_padding": reference_cat_info["total_cardinalities_with_padding"],
            "reference_cat_encoding_stats": reference_cat_info["encoding_stats"],
            "vocab_policy": {
                "padding_id": 0,
                "unk_id": 1,
                "description": "Per-fold and final-model vocabularies are built only on time-visible history. Full sorted raw vocab values are saved inside model checkpoints.",
            },
        },

        "model_config": asdict(model_config),

        "ensemble_variants": ensemble_variant_info,

        "files": {
            "debug_log": str(output_dir / "debug.log"),
            "run_options": str(output_dir / "run_options.json"),
            "results": str(output_dir / "results.json"),
            "oof_predictions": str(output_dir / "oof_predictions.parquet"),
            "fold_metrics_detailed": str(output_dir / "fold_metrics_detailed.json"),
            "test_predictions_cv_ensemble": str(cv_ensemble_pq),
            "submission_cv_ensemble_csv": str(cv_ensemble_csv),
            "test_predictions_final_model": str(final_model_test_path) if final_model_test_path else None,
            "submission_final_model_csv": str(final_model_submission_csv) if final_model_submission_csv else None,
        },

        "fold_summary": [
            pick_keys(fm, fold_summary_keys)
            for fm in fold_metrics_list
        ],
    }

    save_json(output_dir / "results.json", results)

    logger.info("Finished successfully.")
    logger.info("OOF final AP: %s", fmt_optional(oof_metrics["final_ap"]))
    logger.info("OOF red AP: %s", fmt_optional(oof_metrics["red_ap"]))
    logger.info("OOF suspicious AP: %s", fmt_optional(oof_metrics["suspicious_ap"]))
    logger.info("OOF RY AP: %s", fmt_optional(oof_metrics["ry_ap"]))
    logger.info("Mean fold final AP: %s", "nan" if mean_fold_final_ap is None else f"{mean_fold_final_ap:.6f}")
    logger.info("Mean fold red AP: %s", "nan" if mean_fold_red_ap is None else f"{mean_fold_red_ap:.6f}")
    logger.info("Mean fold suspicious AP: %s", "nan" if mean_fold_suspicious_ap is None else f"{mean_fold_suspicious_ap:.6f}")
    logger.info("Mean fold RY AP: %s", "nan" if mean_fold_ry_ap is None else f"{mean_fold_ry_ap:.6f}")
    logger.info("Mean fold ES final AP: %s", "nan" if mean_fold_es_final_ap is None else f"{mean_fold_es_final_ap:.6f}")
    logger.info("Mean fold ES red AP: %s", "nan" if mean_fold_es_red_ap is None else f"{mean_fold_es_red_ap:.6f}")
    logger.info("Mean fold ES suspicious AP: %s", "nan" if mean_fold_es_suspicious_ap is None else f"{mean_fold_es_suspicious_ap:.6f}")
    logger.info("Mean fold ES RY AP: %s", "nan" if mean_fold_es_ry_ap is None else f"{mean_fold_es_ry_ap:.6f}")
    logger.info("Saved CV submission CSV to: %s", cv_ensemble_csv)
    if final_model_submission_csv is not None:
        logger.info("Saved final-model submission CSV to: %s", final_model_submission_csv)
    logger.info("Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
