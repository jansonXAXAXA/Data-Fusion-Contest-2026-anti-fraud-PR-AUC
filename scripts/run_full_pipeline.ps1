# Полный пайплайн: solution1 → solution2 → pipeline1st.ipynb → collect → AGI blend
# Требуется: каталог data/ в корне репо с сырыми данными соревнования (см. README).
$ErrorActionPreference = "Stop"
$PublicRoot = Split-Path -Parent $PSScriptRoot
Set-Location $PublicRoot

if (-not $env:DATA_FUSION_DATA) {
    $env:DATA_FUSION_DATA = Join-Path $PublicRoot "data"
}

Write-Host "== [1/4] solution1: CatBoost + CoLES + refit ==" -ForegroundColor Cyan
Set-Location (Join-Path $PublicRoot "solution1")
python run_catboost.py
python run_coles.py
python run_coles_refit.py
Set-Location $PublicRoot

Write-Host "== [2/4] solution2: подготовка parquet (если ещё нет) и обучение ==" -ForegroundColor Cyan
Set-Location (Join-Path $PublicRoot "solution2")
if (-not (Test-Path (Join-Path $PublicRoot "data\full_dataset.parquet"))) {
    Write-Host "Запустите вручную: python prepare_data.py <raw_dir> ..\data\full_dataset.parquet" -ForegroundColor Yellow
}
python train_last_n_pooling.py `
    --input-path (Join-Path $PublicRoot "data\full_dataset.parquet") `
    --output-dir results `
    --cv-mode sliding `
    --threads 8 `
    --gpu `
    --target1-sample-frac 0.05 `
    --max-epochs 50 `
    --patience 6 `
    --lr 0.0003 `
    --batch-size-gpu 256 `
    --multitask-inference-blend 0 `
    --one-hot-max-cardinality 32 `
    --use-session-branch `
    --use-label-history `
    --use-future-branches
Set-Location $PublicRoot

Write-Host "== [3/4] pipeline1st.ipynb (nbconvert) ==" -ForegroundColor Cyan
python -m jupyter nbconvert --to notebook --execute pipeline1st.ipynb --ExecutePreprocessor.timeout=-1 --inplace

Write-Host "== Сборка входов для AGI + блендинг ==" -ForegroundColor Cyan
python scripts/collect_blend_inputs.py --repo-root $PublicRoot
Set-Location (Join-Path $PublicRoot "AGI")
python blend_submissions.py
Set-Location $PublicRoot
Write-Host "Готово: AGI\sub_totalblend.csv" -ForegroundColor Green
