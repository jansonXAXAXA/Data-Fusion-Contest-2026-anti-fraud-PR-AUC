"""CoLES (Contrastive Learning for Event Sequences).

Обучение на 91M+86M транзакциях -> эмбеддинги клиентов.
Использует pytorch-lifestream.

Пайплайн:
1. Подготовка последовательностей транзакций по клиентам (pretrain+train)
2. Обучение CoLES энкодера (GRU) контрастивным лоссом
3. Извлечение эмбеддингов клиентов (256-мерные)
4. Добавление к фичам CatBoost -> refit -> submission
"""

import gc, sys, time, warnings, functools
print = functools.partial(print, flush=True)
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

warnings.filterwarnings("ignore")

_SOL1 = Path(__file__).resolve().parent
if str(_SOL1) not in sys.path:
    sys.path.insert(0, str(_SOL1))

import logging
from paths import CACHE_COLES, DATA_DIR

_log_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
_logger = logging.getLogger("coles")
_logger.setLevel(logging.INFO)
for h in _logger.handlers[:]: _logger.removeHandler(h)

_fh = logging.FileHandler(CACHE_COLES / "coles.log", mode="w", encoding="utf-8")
_fh.setFormatter(_log_fmt)
_logger.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_log_fmt)
_logger.addHandler(_ch)
log = _logger.info

CACHE_COLES.mkdir(parents=True, exist_ok=True)
SEED = 42

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Шаг 1: Подготовка последовательностей

def prepare_sequences():
    """Загрузка pretrain+train, построение последовательностей по клиентам."""
    log("Подготовка последовательностей...")

    all_events = []

    # Загрузка pretrain (91M) + train (86M) по частям
    for period, files in [
        ("pretrain", [f"pretrain_part_{i}.parquet" for i in [1,2,3]]),
        ("train", [f"train_part_{i}.parquet" for i in [1,2,3]]),
    ]:
        for f in files:
            log(f"  Загрузка {f}...")
            df = pl.read_parquet(DATA_DIR / f, columns=[
                "customer_id", "event_dttm", "event_type_nm", "event_desc",
                "channel_indicator_type", "channel_indicator_sub_type",
                "operaton_amt", "mcc_code", "pos_cd", "timezone",
                "operating_system_type", "phone_voip_call_state",
                "web_rdp_connection",
            ])
            df = df.with_columns([
                pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("dt"),
                pl.col("operaton_amt").cast(pl.Float32).fill_null(0.0).alias("amt"),
                pl.col("event_type_nm").cast(pl.Int16, strict=False).fill_null(0).alias("etype"),
                pl.col("event_desc").cast(pl.Int16, strict=False).fill_null(0).alias("edesc"),
                pl.col("channel_indicator_type").cast(pl.Int8, strict=False).fill_null(0).alias("ch_type"),
                pl.col("channel_indicator_sub_type").cast(pl.Int8, strict=False).fill_null(0).alias("ch_sub"),
                pl.col("mcc_code").cast(pl.Int16, strict=False).fill_null(0).alias("mcc"),
                pl.col("pos_cd").cast(pl.Int8, strict=False).fill_null(0).alias("pos"),
                pl.col("timezone").cast(pl.Int16, strict=False).fill_null(0).alias("tz"),
                pl.col("operating_system_type").cast(pl.Int8, strict=False).fill_null(0).alias("os"),
                pl.col("phone_voip_call_state").cast(pl.Int8, strict=False).fill_null(0).alias("voip"),
                pl.col("web_rdp_connection").cast(pl.Int8, strict=False).fill_null(0).alias("rdp"),
            ]).select(["customer_id", "dt", "amt", "etype", "edesc", "ch_type", "ch_sub",
                       "mcc", "pos", "tz", "os", "voip", "rdp"])
            all_events.append(df)
            del df; gc.collect()

    log("  Конкатенация...")
    all_df = pl.concat(all_events, how="vertical_relaxed")
    del all_events; gc.collect()

    log(f"  Всего событий: {all_df.height:,}")

    # Сортировка по клиенту + времени
    all_df = all_df.sort(["customer_id", "dt"])

    # Временные фичи
    all_df = all_df.with_columns([
        pl.col("dt").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("dt").dt.weekday().cast(pl.Int8).alias("dow"),
        pl.col("amt").abs().log1p().cast(pl.Float32).alias("amt_log"),
    ])

    # Группировка по клиентам
    log("  Группировка по клиентам...")
    # Берём последние 500 событий на клиента (для экономии памяти)
    grouped = all_df.group_by("customer_id").agg([
        pl.col("etype").tail(500).alias("etype_seq"),
        pl.col("edesc").tail(500).alias("edesc_seq"),
        pl.col("ch_type").tail(500).alias("ch_type_seq"),
        pl.col("ch_sub").tail(500).alias("ch_sub_seq"),
        pl.col("mcc").tail(500).alias("mcc_seq"),
        pl.col("pos").tail(500).alias("pos_seq"),
        pl.col("tz").tail(500).alias("tz_seq"),
        pl.col("os").tail(500).alias("os_seq"),
        pl.col("voip").tail(500).alias("voip_seq"),
        pl.col("rdp").tail(500).alias("rdp_seq"),
        pl.col("hour").tail(500).alias("hour_seq"),
        pl.col("dow").tail(500).alias("dow_seq"),
        pl.col("amt_log").tail(500).alias("amt_log_seq"),
        pl.len().alias("total_events"),
    ])

    del all_df; gc.collect()
    log(f"  {grouped.height} клиентов, сохранение...")
    grouped.write_parquet(CACHE_COLES / "customer_sequences.parquet")
    return grouped


# Шаг 2: PyTorch Dataset + Модель

CAT_FEATURES = ["etype", "edesc", "ch_type", "ch_sub", "mcc", "pos", "tz", "os", "voip", "rdp", "hour", "dow"]
NUM_FEATURES = ["amt_log"]
# Максимальная кардинальность для каждой категориальной фичи (для таблиц эмбеддингов)
CAT_CARDS = {"etype": 20, "edesc": 130, "ch_type": 10, "ch_sub": 20, "mcc": 25,
             "pos": 25, "tz": 60, "os": 15, "voip": 5, "rdp": 5, "hour": 25, "dow": 8}
EMB_DIM = 8  # на каждую категорию


class CustomerSeqDataset(Dataset):
    """Датасет для контрастивного обучения. Возвращает две случайные подпоследовательности на клиента."""
    def __init__(self, sequences_df, seq_len=64):
        self.customers = sequences_df["customer_id"].to_list()
        self.cat_seqs = {}
        for c in CAT_FEATURES:
            self.cat_seqs[c] = sequences_df[f"{c}_seq"].to_list()
        self.num_seqs = {}
        for c in NUM_FEATURES:
            self.num_seqs[c] = sequences_df[f"{c}_seq"].to_list()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.customers)

    def _get_subsequence(self, idx, rng):
        """Получить случайную подпоследовательность длины seq_len."""
        full_len = len(self.cat_seqs[CAT_FEATURES[0]][idx])
        if full_len <= self.seq_len:
            start = 0
            end = full_len
        else:
            start = rng.randint(0, full_len - self.seq_len)
            end = start + self.seq_len

        cat_data = []
        for c in CAT_FEATURES:
            vals = self.cat_seqs[c][idx][start:end]
            arr = np.array(vals, dtype=np.int64)
            arr = np.clip(arr + 1, 0, CAT_CARDS[c] - 1)  # сдвиг на 1, обрезка
            cat_data.append(arr)

        num_data = []
        for c in NUM_FEATURES:
            vals = self.num_seqs[c][idx][start:end]
            num_data.append(np.array(vals, dtype=np.float32))

        return np.stack(cat_data, axis=1), np.stack(num_data, axis=1), end - start

    def __getitem__(self, idx):
        rng = np.random.RandomState()
        cat1, num1, len1 = self._get_subsequence(idx, rng)
        cat2, num2, len2 = self._get_subsequence(idx, rng)
        return cat1, num1, len1, cat2, num2, len2


def collate_fn(batch):
    """Паддинг последовательностей до максимальной длины в батче."""
    max_len = max(max(b[2], b[5]) for b in batch)
    n_cat = len(CAT_FEATURES)
    n_num = len(NUM_FEATURES)
    bs = len(batch)

    cat1 = torch.zeros(bs, max_len, n_cat, dtype=torch.long)
    num1 = torch.zeros(bs, max_len, n_num, dtype=torch.float32)
    mask1 = torch.zeros(bs, max_len, dtype=torch.bool)
    cat2 = torch.zeros(bs, max_len, n_cat, dtype=torch.long)
    num2 = torch.zeros(bs, max_len, n_num, dtype=torch.float32)
    mask2 = torch.zeros(bs, max_len, dtype=torch.bool)

    for i, (c1, n1, l1, c2, n2, l2) in enumerate(batch):
        cat1[i, :l1] = torch.from_numpy(c1)
        num1[i, :l1] = torch.from_numpy(n1)
        mask1[i, :l1] = True
        cat2[i, :l2] = torch.from_numpy(c2)
        num2[i, :l2] = torch.from_numpy(n2)
        mask2[i, :l2] = True

    return cat1, num1, mask1, cat2, num2, mask2


class CoLESEncoder(nn.Module):
    """Энкодер последовательностей транзакций: эмбеддинги + GRU."""
    def __init__(self, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            c: nn.Embedding(CAT_CARDS[c], EMB_DIM, padding_idx=0)
            for c in CAT_FEATURES
        })
        input_dim = len(CAT_FEATURES) * EMB_DIM + len(NUM_FEATURES)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.proj = nn.Linear(hidden_dim, hidden_dim)  # проекционная голова
        self.hidden_dim = hidden_dim

    def forward(self, cat, num, mask):
        # Эмбеддинги категориальных фичей
        embs = []
        for i, c in enumerate(CAT_FEATURES):
            embs.append(self.embeddings[c](cat[:, :, i]))
        embs.append(num)
        x = torch.cat(embs, dim=-1)  # (B, T, input_dim)

        # GRU
        output, _ = self.gru(x)  # (B, T, hidden)

        # Mean pooling с маской
        mask_exp = mask.unsqueeze(-1).float()  # (B, T, 1)
        pooled = (output * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)  # (B, hidden)

        # Проекция
        z = self.proj(pooled)
        return z


class CoLESLoss(nn.Module):
    """NT-Xent контрастивный лосс."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Нормализация
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Матрица схожести
        bs = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Маскируем само-схожесть
        mask = torch.eye(2 * bs, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # Положительные пары: (i, i+B) и (i+B, i)
        labels = torch.cat([torch.arange(bs, 2*bs), torch.arange(0, bs)]).to(z.device)

        loss = nn.functional.cross_entropy(sim, labels)
        return loss


# Шаг 3: Обучение

def train_coles(sequences_df, epochs=15, hidden_dim=256, batch_size=256, lr=1e-3):
    """Обучение CoLES энкодера."""
    log(f"Обучение CoLES: epochs={epochs}, hidden={hidden_dim}, bs={batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Устройство: {device}")

    dataset = CustomerSeqDataset(sequences_df, seq_len=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=0, collate_fn=collate_fn, drop_last=True)

    model = CoLESEncoder(hidden_dim=hidden_dim).to(device)
    criterion = CoLESLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Параметров модели: {n_params:,}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for cat1, num1, mask1, cat2, num2, mask2 in loader:
            cat1, num1, mask1 = cat1.to(device), num1.to(device), mask1.to(device)
            cat2, num2, mask2 = cat2.to(device), num2.to(device), mask2.to(device)

            z1 = model(cat1, num1, mask1)
            z2 = model(cat2, num2, mask2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        log(f"  Эпоха {epoch+1}/{epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    return model


# Шаг 4: Извлечение эмбеддингов

def extract_embeddings(model, sequences_df, batch_size=512):
    """Извлечение клиентских эмбеддингов из обученного энкодера."""
    log("Извлечение эмбеддингов...")
    device = next(model.parameters()).device
    model.eval()

    dataset = CustomerSeqDataset(sequences_df, seq_len=500)  # полная последовательность
    all_embeddings = []
    all_customer_ids = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_idx = list(range(i, min(i + batch_size, len(dataset))))
            batch = [dataset[j] for j in batch_idx]
            # Используем первую подпоследовательность (полную)
            cat1, num1, mask1, _, _, _ = collate_fn(batch)
            cat1, num1, mask1 = cat1.to(device), num1.to(device), mask1.to(device)

            z = model(cat1, num1, mask1)
            all_embeddings.append(z.cpu().numpy())
            all_customer_ids.extend([dataset.customers[j] for j in batch_idx])

            if (i // batch_size) % 50 == 0:
                log(f"  {i}/{len(dataset)} клиентов...")

    embeddings = np.concatenate(all_embeddings, axis=0)
    log(f"  Размер эмбеддингов: {embeddings.shape}")

    # Сохраняем как polars DataFrame
    emb_data = {"customer_id": all_customer_ids}
    for d in range(embeddings.shape[1]):
        emb_data[f"coles_{d}"] = embeddings[:, d].astype(np.float32)

    emb_df = pl.DataFrame(emb_data)
    emb_df.write_parquet(CACHE_COLES / "coles_embeddings.parquet")
    log(f"  Сохранено: {emb_df.shape}")
    return emb_df


def main():
    t0 = time.time()
    log("=" * 60)
    log("CoLES ПАЙПЛАЙН ПРЕДОБУЧЕНИЯ")
    log("=" * 60)

    # Шаг 1: Подготовка последовательностей
    seq_path = CACHE_COLES / "customer_sequences.parquet"
    if seq_path.exists():
        log("Загрузка кэшированных последовательностей...")
        sequences = pl.read_parquet(seq_path)
    else:
        sequences = prepare_sequences()

    log(f"Последовательностей: {sequences.height} клиентов")

    # Шаг 2: Обучение CoLES
    model = train_coles(sequences, epochs=15, hidden_dim=256, batch_size=256, lr=1e-3)

    # Шаг 3: Сохранение модели
    torch.save(model.state_dict(), CACHE_COLES / "coles_model.pt")
    log("Модель сохранена")

    # Шаг 4: Извлечение эмбеддингов
    emb_df = extract_embeddings(model, sequences)

    log(f"\nОбщее время: {(time.time()-t0)/60:.1f} мин")
    log("ГОТОВО! Далее: добавить coles_embeddings.parquet к фичам CatBoost")


if __name__ == "__main__":
    main()
