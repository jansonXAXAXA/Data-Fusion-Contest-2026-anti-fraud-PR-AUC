#!/usr/bin/env python3
"""Копирует сабмиты из solution1, solution2 и cache ноутбука в AGI/ под именами для блендинга."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Корень репозитория (PUBLIC)",
    )
    args = ap.parse_args()
    root = args.repo_root.resolve()
    agi = root / "AGI"

    mappings: list[tuple[Path, Path]] = [
        (
            root / "solution1" / "submissions" / "coles_seed_fb50.csv",
            agi / "submission_ICEQ_PUBLIC.csv",
        ),
        (
            root / "solution2" / "results" / "submission_cv_ensemble.csv",
            agi / "submission_DL_PUBLIC.csv",
        ),
        (
            root / "cache" / "submission_MINE.csv",
            agi / "submission_MINE.csv",
        ),
    ]

    agi.mkdir(parents=True, exist_ok=True)
    for src, dst in mappings:
        if not src.is_file():
            raise FileNotFoundError(f"Нет файла: {src}\nСначала выполните соответствующий шаг пайплайна.")
        shutil.copy2(src, dst)
        print(f"Copied {src.name} → {dst}")


if __name__ == "__main__":
    main()
