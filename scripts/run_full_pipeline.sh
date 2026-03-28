#!/usr/bin/env bash
# Полный пайплайн: solution1 → solution2 → pipeline1st.ipynb → collect → AGI blend
set -euo pipefail
PUBLIC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PUBLIC_ROOT"
export DATA_FUSION_DATA="${DATA_FUSION_DATA:-$PUBLIC_ROOT/data}"

echo "== [1/4] solution1: CatBoost + CoLES + refit =="
cd "$PUBLIC_ROOT/solution1"
python run_catboost.py
python run_coles.py
python run_coles_refit.py
cd "$PUBLIC_ROOT"

echo "== [2/4] solution2: обучение (нужен data/full_dataset.parquet) =="
cd "$PUBLIC_ROOT/solution2"
if [[ ! -f "$PUBLIC_ROOT/data/full_dataset.parquet" ]]; then
  echo "Создайте parquet: python prepare_data.py <raw_dir> $PUBLIC_ROOT/data/full_dataset.parquet" >&2
fi
python train_last_n_pooling.py \
  --input-path "$PUBLIC_ROOT/data/full_dataset.parquet" \
  --output-dir results \
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
cd "$PUBLIC_ROOT"

echo "== [3/4] pipeline1st.ipynb =="
python -m jupyter nbconvert --to notebook --execute pipeline1st.ipynb --ExecutePreprocessor.timeout=-1 --inplace

echo "== Сборка входов для AGI + блендинг =="
python scripts/collect_blend_inputs.py --repo-root "$PUBLIC_ROOT"
cd "$PUBLIC_ROOT/AGI"
python blend_submissions.py
cd "$PUBLIC_ROOT"
echo "Готово: AGI/sub_totalblend.csv"
