# MASTER Stock Prediction Model — Local Qlib Data

This project runs the **MASTER** model (AAAI-2024) on your own local Chinese A-share stock data stored in qlib format.

> **What is MASTER?**
> MASTER (Market-Guided Stock Transformer) is a deep learning model for stock return prediction published at AAAI 2024. It uses temporal self-attention (across days) and spatial self-attention (across stocks) to jointly model how stocks move together over time.
> Original paper and code: [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER)

> **What is qlib?**
> [Qlib](https://github.com/microsoft/qlib) is Microsoft's open-source quantitative investment platform. It standardises how stock data is stored (binary `.bin` files) and provides tools for data loading, feature engineering, and model training.

---

## What this project does

1. Loads your local qlib stock data (OHLCV binary files)
2. Computes 10 financial features from the raw price/volume fields
3. Trains the MASTER model to predict each stock's next-day return
4. Reports IC, ICIR, RankIC, RankICIR on the validation set each epoch
5. Saves a model checkpoint and a CSV of predictions on the test set

---

## Repository files

| File | Role |
|------|------|
| `workflow_config_master.yaml` | **Edit this first** — all settings: data path, date ranges, instrument universe, model hyperparameters |
| `run_qlib.py` | **Run this** — reads the config, trains the model, saves predictions |
| `master_model.py` | MASTER neural network definition and training logic |
| `base_model.py` | Shared training utilities (loss, sampler, metrics) |
| `requirements.txt` | Python package dependencies |

---

## Requirements

- Python 3.8+
- A local qlib data folder (see structure below)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Data folder structure

Your qlib data folder must look like this:

```
my_qlib_data/
├── calendars/
│   └── day.txt              # one trading date per line, e.g.: 2020-01-02
├── features/
│   └── sh600000/            # one sub-folder per stock (e.g. sh600000, sz000001, bj430090)
│       ├── close.day.bin
│       ├── open.day.bin
│       ├── high.day.bin
│       ├── low.day.bin
│       ├── volume.day.bin
│       ├── amount.day.bin
│       ├── vwap.day.bin
│       ├── change.day.bin
│       ├── factor.day.bin
│       └── adjclose.day.bin
└── instruments/
    ├── all.txt
    ├── csi300.txt           # which stocks are in CSI 300 and when
    ├── csi500.txt
    ├── csi800.txt
    └── ...
```

**Instruments file format** — tab-separated, no header, one stock per line:
```
SH600000    2020-01-02    2026-03-20
SZ000001    2020-01-02    2026-03-20
BJ430090    2021-11-15    2026-03-20
```
The symbol prefix (`SH` / `SZ` / `BJ`) must match the folder name case in `features/`.

---

## Quick start

**Step 1 — Edit the config file**

Open `workflow_config_master.yaml` and set:
- `provider_uri`: path to your qlib data folder
- `start_time` / `end_time`: full date range of your data
- `segments`: train / valid / test split dates
- `market`: instrument universe (e.g. `csi300`, `csi500`, `all`)

**Step 2 — Run**

```bash
python run_qlib.py --config workflow_config_master.yaml
```

That's it. The script will print training progress, then save:
- `model/` — the best model checkpoint (`.pkl`)
- `predictions.csv` — model scores for every stock on every test day

You can override any config value from the command line without editing the YAML:
```bash
python run_qlib.py --config workflow_config_master.yaml --n_epochs 20 --instrument csi500
```

---

## Training output explained

Each epoch prints:
```
Epoch 15, train_loss 0.998551, val_loss 0.999919 | IC 0.0312, ICIR 0.421, RankIC 0.0287, RankICIR 0.398
```

| Column | Meaning |
|--------|---------|
| `train_loss` | MSE loss on training data (z-scored labels, so ~1.0 at the start is normal) |
| `val_loss` | Same loss on the validation set |
| `IC` | Information Coefficient — daily Pearson correlation between predicted scores and actual next-day returns, averaged across all validation days. Higher is better. Meaningful at > 0.02, strong at > 0.05 |
| `ICIR` | IC ÷ std(IC) — how consistent the IC is day-to-day (like a Sharpe ratio). Good at > 0.3 |
| `RankIC` | Same as IC but using rank-correlation (Spearman), more robust to outliers |
| `RankICIR` | Consistency of RankIC |

Early epochs will show IC near 0. You expect it to start rising after ~10 epochs.

---

## Predictions CSV explained

`predictions.csv` contains the model's output score for every stock on every day in the test period.

```
datetime,instrument,0
2025-01-02,SH600000,0.0452
2025-01-02,SZ000001,-0.0231
2025-01-02,BJ430090,0.0187
...
2026-03-20,SH600000,0.0318   ← last trading day, first stock
2026-03-20,SZ000001,-0.0089
2026-03-20,BJ430090,0.0201   ← last row: last stock on the last test day
```

- **Each row** = one stock on one trading day
- **The score** is a relative ranking signal, not a raw return prediction. A higher score means the model thinks this stock will outperform the others on the next trading day
- **How to use it**: rank all stocks by score on a given date — top-ranked stocks are the model's buy candidates for that day
- **The last row** has no special meaning; it is simply the last stock alphabetically on the last date in the test set

---

## Run modes

| Mode | Command | Use case |
|------|---------|----------|
| Train + predict (default) | `--mode both` | Full pipeline: train from scratch, then evaluate |
| Train only | `--mode train` | Train and save checkpoint, skip prediction |
| Predict only | `--mode predict` | Load a saved checkpoint and generate predictions |

Predict-only example (after training):
```bash
python run_qlib.py --config workflow_config_master.yaml \
    --mode predict \
    --model_path model/csi300master_0.pkl \
    --output_csv predictions.csv
```

---

## Features

The model uses 10 features computed from your raw price/volume data:

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `OPEN_RET` | `open / prev_close - 1` | Overnight gap |
| `HIGH_RET` | `high / prev_close - 1` | Intraday upside |
| `LOW_RET` | `low / prev_close - 1` | Intraday downside |
| `CLOSE_RET` | `close / prev_close - 1` | Daily return |
| `VWAP_RET` | `vwap / prev_close - 1` | Volume-weighted return |
| `LOG_VOL` | `log(volume + 1)` | Trading activity |
| `LOG_AMT` | `log(amount + 1)` | Trading value |
| `INTRADAY` | `(close - low) / (high - low)` | Where price closed within the day's range |
| `MOM5` | `close[-1] / close[-6] - 1` | 5-day momentum |
| `MOM10` | `close[-1] / close[-11] - 1` | 10-day momentum |

**Label**: next trading day's return (`close[+1] / close[0] - 1`)

All features are normalised using `RobustZScoreNorm` (clip outliers at ±3σ) fitted on the training period, then applied consistently to validation and test.

---

## Model architecture

```
Input: (N stocks, T=8 days, 10 features)
    │
    ├─ Linear projection       10 → 256 (d_model)
    ├─ Positional Encoding     adds day-position signal
    ├─ Temporal Attention      each stock learns from its own past 8 days
    ├─ Spatial Attention       stocks learn from each other (cross-sectional)
    ├─ Temporal Aggregation    compress 8 time steps → 1 vector
    └─ Linear decoder          256 → 1 score per stock
```

The original MASTER also has a market-information Gate (uses CSI300 index features to weight the 158 Alpha158 stock factors). Since our data does not include a separate market index, the Gate is disabled and all 10 features are used directly.

---

## Adapations from the original repo

This project is based on `qlib-update/pytorch_master_ts.py` from the original MASTER repo, with these changes:

1. **Absolute imports** — changed `from ...data.dataset import DatasetH` to `from qlib.data.dataset import DatasetH` so the file works as a standalone script without being installed inside the qlib package.

2. **No-gate mode** — the Gate is skipped when `gate_input_start_index == gate_input_end_index` (our default), since we have no market-index features.

3. **Validation bug fix** (from qlib-update) — validation now correctly uses `infer_processors` (`DK_I`) to apply normalisation statistics fitted only on training data, preventing data leakage.

4. **10-feature handler** — replaced `Alpha158` (158 engineered factors) with a lightweight `DataHandlerLP` using 10 OHLCV-derived features that work with any standard qlib binary dataset.

5. **IC/ICIR/RankIC/RankICIR logged each epoch** — more informative than MSE loss alone for stock prediction tasks.
