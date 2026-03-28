"""Рефит пайплайна с CoLES эмбеддингами -> сабмиты.

Читает фичи из cache/ (тот же каталог, что и run_catboost.py), v5_config.json.
Присоединяет CoLES эмбеддинги, рефитит 4 модели по 3 сидам, создает файл сабмита.
"""
import gc, sys, time, warnings, functools, json
import numpy as np, pandas as pd, polars as pl
from pathlib import Path
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings("ignore")

_SOL1 = Path(__file__).resolve().parent
if str(_SOL1) not in sys.path:
    sys.path.insert(0, str(_SOL1))

log = functools.partial(print, flush=True)

# ── Конфигурация ──
from paths import CACHE_DIR, CACHE_COLES, DATA_DIR, SUBMISSIONS

CACHE_V5 = CACHE_DIR
SEED = 42

# Загружаем подобранный конфиг из v5
with open(CACHE_V5 / "v5_config.json") as f:
    cfg = json.load(f)
BEST_CB_W = tuple(cfg["best_w"])
BEST_ALPHA = cfg["best_alpha"]
BI_MAIN = cfg["bi_main"]
BI_SUSP = cfg["bi_susp"]
BI_RG = cfg["bi_rg"]
BI_FB = cfg["bi_fb"]
log(f"Config: CB_W={BEST_CB_W}, alpha={BEST_ALPHA}, BI_MAIN={BI_MAIN}, BI_SUSP={BI_SUSP}, BI_RG={BI_RG}, BI_FB={BI_FB}")

CAT_COLS = ["customer_id","event_type_nm","event_desc","channel_indicator_type","channel_indicator_sub_type","currency_iso_cd","mcc_code_i","pos_cd","timezone","operating_system_type","phone_voip_call_state","web_rdp_connection","developer_tools_i","compromised_i","prev_mcc_code_i"]
FB_FEATURE_COLS = ["cust_prev_red_lbl_cnt","cust_prev_yellow_lbl_cnt","cust_prev_labeled_cnt","cust_prev_red_lbl_rate","cust_prev_yellow_lbl_rate","cust_prev_susp_lbl_rate","cust_prev_any_red_flag","cust_prev_any_yellow_flag","sec_since_prev_red_lbl","sec_since_prev_yellow_lbl","cnt_prev_labeled_same_desc","cnt_prev_red_same_desc_lbl","cnt_prev_yellow_same_desc_lbl","red_rate_prev_same_desc_lbl"]
META_COLS = ["event_id","period","event_ts","is_train_sample","is_test","train_target_raw","target_bin"]

def _sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-40,40)))
def _logit(p): p=np.clip(p,1e-8,1-1e-8); return np.log(p/(1-p))
def make_weights(r): return np.where(r==1,10.0,np.where(r==0,2.5,1.0)).astype(np.float32)
def refit(X,y,w,cats,iters,lr=0.05,d=8,l2=None,seed=42):
    p={"iterations":max(300,iters),"learning_rate":lr,"depth":d,"loss_function":"Logloss","random_seed":seed,"task_type":"GPU","devices":"0","allow_writing_files":False,"verbose":500}
    if l2: p["l2_leaf_reg"]=l2
    m=CatBoostClassifier(**p); m.fit(Pool(X,np.asarray(y).ravel(),weight=np.asarray(w,dtype=np.float32),cat_features=cats),verbose=500); return m

t0=time.time()

# ── Загрузка v5 фичей (уже есть приоры, профили) ──
log("Загрузка v5 фичей из cache_v5...")
features = pl.concat([
    pl.read_parquet(CACHE_V5 / f"features_part_{i}.parquet")
    for i in [1, 2, 3]
], how="vertical_relaxed").unique(subset=["event_id"])
log(f"  Сырые фичи: {features.shape}")

# Присоединяем CoLES эмбеддинги
log("Присоединение CoLES эмбеддингов...")
coles = pl.read_parquet(CACHE_COLES / "coles_embeddings.parquet")
features = features.join(coles, on="customer_id", how="left")
cc = [c for c in coles.columns if c != "customer_id"]
features = features.with_columns([pl.col(c).fill_null(0.0) for c in cc])
log(f"  +{len(cc)} CoLES измерений, итого: {features.shape}")
del coles; gc.collect()

# ── Априоры (считаем на ВСЁМ трейне — корректно для рефита, без val-сплита) ──
log("Построение априоров (весь трейн)...")
labels_lf = pl.scan_parquet(DATA_DIR / "train_labels.parquet")
PC = {
    "event_desc": pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1).alias("event_desc"),
    "mcc_code_i": pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),
    "event_type_nm": pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1).alias("event_type_nm"),
    "pos_cd": pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("pos_cd"),
}
pfc = []
for k, e in PC.items():
    lf = pl.concat([
        pl.scan_parquet(DATA_DIR / f"train_part_{i}.parquet").select([pl.col("event_id"), e])
        for i in [1, 2, 3]
    ], how="vertical_relaxed")
    t2 = lf.group_by(k).len().rename({"len": f"prior_{k}_cnt"})
    lb = lf.join(labels_lf, on="event_id", how="inner").group_by(k).agg([
        pl.len().alias("_l"), pl.sum("target").cast(pl.Float64).alias("_r")
    ])
    pr = t2.join(lb, on=k, how="left").with_columns([
        pl.col("_l").fill_null(0.0), pl.col("_r").fill_null(0.0)
    ]).with_columns([
        ((pl.col("_r") + 1) / (pl.col(f"prior_{k}_cnt") + 200)).cast(pl.Float32).alias(f"prior_{k}_red_rate"),
        ((pl.col("_r") + 1) / (pl.col("_l") + 2)).cast(pl.Float32).alias(f"prior_{k}_red_share"),
    ]).select([k, f"prior_{k}_cnt", f"prior_{k}_red_rate", f"prior_{k}_red_share"]).collect()
    features = features.join(pr, on=k, how="left")
    pfc.extend([c for c in pr.columns if c != k])
    log(f"  {k}: {pr.height} категорий")
if pfc:
    features = features.with_columns([pl.col(c).fill_null(pl.col(c).mean()).alias(c) for c in pfc])

# ── Конвертация в pandas ──
log("Конвертация в pandas...")
train_pl = features.filter(
    pl.col("is_train_sample") & ((pl.col("train_target_raw") != -1) | (pl.col("event_id") % 3 == 0))
).with_columns(pl.col("target_bin").cast(pl.Int8))
test_pl = features.filter(pl.col("is_test"))
del features; gc.collect()
train_df = train_pl.to_pandas(); test_df = test_pl.to_pandas()
del train_pl, test_pl; gc.collect()
for d in [train_df, test_df]:
    d["event_ts"] = pd.to_datetime(d["event_ts"])

feature_cols = [c for c in train_df.columns if c not in META_COLS and c not in ["target", "keep_green", "event_date"] and c not in FB_FEATURE_COLS]
fb_cols = [c for c in FB_FEATURE_COLS if c in train_df.columns]
fb_feature_cols = feature_cols + fb_cols
cat_cols = [c for c in CAT_COLS if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]
for d in [train_df, test_df]:
    for c in cat_cols: d[c] = d[c].fillna(-1).astype(np.int64)
med = train_df[num_cols].median(numeric_only=True)
for d in [train_df, test_df]:
    d[num_cols] = d[num_cols].fillna(med)
    for c in fb_cols:
        if c in d.columns: d[c] = d[c].fillna(0.0).astype(np.float32)
    for c in num_cols:
        if d[c].dtype == np.float64: d[c] = d[c].astype(np.float32)
train_df = train_df.sort_values("event_ts").reset_index(drop=True)

raw = train_df["train_target_raw"].values
y = train_df["target_bin"].astype(np.int8).values
y_susp = (raw != -1).astype(np.int8)
w = make_weights(raw)
w_susp = np.where(raw != -1, 6.0, 1.2).astype(np.float32)
labeled_mask = raw != -1
log(f"Трейн: {len(train_df):,}, Тест: {len(test_df):,}, Фичей: {len(feature_cols)}")

# ── Рефит по всем сидам ──
wm, wr, wp = BEST_CB_W
log(f"Веса блендинга: MAIN={wm}, REC={wr}, PROD={wp}")

for seed in [42, 123, 777]:
    log(f"\n=== СИД {seed} ===")
    log("MAIN..."); mf_main = refit(train_df[feature_cols], y, w, cat_cols, BI_MAIN, seed=seed); gc.collect()
    log("SUSP..."); mf_susp = refit(train_df[feature_cols], y_susp, w_susp, cat_cols, BI_SUSP, seed=seed); gc.collect()
    log("RG..."); mf_rg = refit(
        train_df.loc[labeled_mask, feature_cols], y[labeled_mask],
        np.where(raw[labeled_mask] == 1, 2.2, 1.0).astype(np.float32),
        cat_cols, BI_RG, lr=0.01, d=4, l2=5, seed=seed
    ); gc.collect()
    log("FB..."); mf_fb = refit(train_df[fb_feature_cols], y, w, cat_cols, BI_FB, seed=seed); gc.collect()

    tpool = Pool(test_df[feature_cols], cat_features=cat_cols)
    t_main = mf_main.predict(tpool, prediction_type="RawFormulaVal")
    t_susp = mf_susp.predict(tpool, prediction_type="RawFormulaVal")
    t_rg = mf_rg.predict(tpool, prediction_type="RawFormulaVal")
    t_prod = _logit(_sigmoid(t_susp) * _sigmoid(t_rg))
    t_cb = wm * t_main + wp * t_prod  # wr=0 обычно, RECENT пропущен
    fb_tpool = Pool(test_df[fb_feature_cols], cat_features=cat_cols)
    t_fb = mf_fb.predict(fb_tpool, prediction_type="RawFormulaVal")

    if seed == 42:
        all_cb = [t_cb]; all_fb = [t_fb]
    else:
        all_cb.append(t_cb); all_fb.append(t_fb)
    del mf_main, mf_susp, mf_rg, mf_fb; gc.collect()

# ── Усреднение сидов + FB инъекция ──
avg_cb = np.mean(all_cb, axis=0)
avg_fb = np.mean(all_fb, axis=0)
labels_df = pl.read_parquet(DATA_DIR / "train_labels.parquet")
labeled_custs = set(labels_df["customer_id"].to_list())
test_has_hist = test_df["customer_id"].isin(labeled_custs).values
rc = rankdata(avg_cb) / len(avg_cb)
r_fb = rankdata(avg_fb) / len(avg_fb)
sample = pd.read_csv(DATA_DIR / "sample_submit.csv")
eids = test_df["event_id"].values

log(f"\nТест has_hist: {test_has_hist.sum()}/{len(test_has_hist)} ({test_has_hist.mean()*100:.1f}%)")

for alpha in [0.3, 0.5, 0.7]:
    bl = rc.copy()
    bl[test_has_hist] = (1 - alpha) * rc[test_has_hist] + alpha * r_fb[test_has_hist]
    sub = sample[["event_id"]].merge(
        pd.DataFrame({"event_id": eids, "predict": bl}), on="event_id", how="left"
    )
    assert sub["predict"].isna().sum() == 0
    sub.to_csv(SUBMISSIONS / f"coles_seed_fb{int(alpha*100)}.csv", index=False)
    log(f"coles_seed_fb{int(alpha*100)}.csv сохранён (alpha={alpha})")

log(f"\nИТОГО: {(time.time()-t0)/60:.1f} мин")
log("ГОТОВО!")
