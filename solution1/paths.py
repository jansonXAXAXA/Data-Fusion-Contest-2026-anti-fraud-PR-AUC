"""Пути для локального запуска и Kaggle.

Переопределение через переменные окружения:
  DATA_FUSION_DATA          — каталог с parquet (train_part_*, pretrain_*, sample_submit.csv, …)
  DATA_FUSION_CACHE         — кеш CatBoost (features_part_*, v5_config.json)
  DATA_FUSION_CACHE_COLES   — кеш CoLES
  DATA_FUSION_SUBMISSIONS   — CSV сабмитов solution1
"""
from __future__ import annotations

import os
from pathlib import Path

SOL1 = Path(__file__).resolve().parent
PUBLIC = SOL1.parent

DATA_DIR = Path(os.environ.get("DATA_FUSION_DATA", PUBLIC / "data"))
CACHE_DIR = Path(os.environ.get("DATA_FUSION_CACHE", SOL1 / "cache"))
CACHE_COLES = Path(os.environ.get("DATA_FUSION_CACHE_COLES", SOL1 / "cache_coles"))
SUBMISSIONS = Path(os.environ.get("DATA_FUSION_SUBMISSIONS", SOL1 / "submissions"))

for _p in (CACHE_DIR, CACHE_COLES, SUBMISSIONS):
    _p.mkdir(parents=True, exist_ok=True)
