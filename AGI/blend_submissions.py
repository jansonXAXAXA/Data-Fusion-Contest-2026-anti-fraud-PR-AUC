#!/usr/bin/env python3
"""Тот же блендинг, что в SMART_AGI.ipynb: z-score логитов → взвешенная смесь → sigmoid.

Запуск из каталога AGI (или с --work-dir):
  python blend_submissions.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def blend_two_csvs(
    path_a: Path,
    path_b: Path,
    out_path: Path,
    weight_a: float,
    eps: float = 1e-7,
) -> pd.DataFrame:
    a = pd.read_csv(path_a).rename(columns={"predict": "pred_a"})
    b = pd.read_csv(path_b).rename(columns={"predict": "pred_b"})
    df = a.merge(b, on="event_id", how="inner")
    za = zscore(logit(df["pred_a"].values, eps))
    zb = zscore(logit(df["pred_b"].values, eps))
    z = weight_a * za + (1.0 - weight_a) * zb
    out = df[["event_id"]].copy()
    out["predict"] = sigmoid(z)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"OK → {out_path}  ({len(out):,} rows)")
    return out


BLEND_STEPS = [
    ("submission_DL_PUBLIC.csv", "submission_ICEQ_PUBLIC.csv", "BLEND1.csv", 0.55, 1e-7),
    ("BLEND1.csv", "submission_MINE.csv", "BLEND2.csv", 0.55, 1e-7),
    ("BLEND2.csv", "BLEND1.csv", "BLEND4.csv", 0.5485, 1e-8),
    ("BLEND1.csv", "BLEND4.csv", "sub_totalblend.csv", 0.541415926, 1e-9),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Blend ICEQ + DL + MINE submissions (logit z-mix).")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory with submission_*.csv files (default: this folder)",
    )
    args = p.parse_args()
    wd = args.work_dir.resolve()
    for pa, pb, outp, w, ep in BLEND_STEPS:
        blend_two_csvs(wd / pa, wd / pb, wd / outp, w, ep)


if __name__ == "__main__":
    main()
