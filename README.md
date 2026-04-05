# MASTER Stock Prediction Model — qlib Workflow

This project runs the **MASTER** model (AAAI-2024) on local Chinese A-share stock data stored in qlib format, using the official qlib workflow for training, signal evaluation, and portfolio backtesting.

> **What is MASTER?**  
> MASTER (Market-Guided Stock Transformer) is a deep learning model for stock return prediction published at AAAI 2024. It uses temporal self-attention (within each stock across days) and spatial self-attention (across stocks within each day) to jointly model cross-sectional stock dynamics.  
> Original paper and code: [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER)

> **What is qlib?**  
> [Qlib](https://github.com/microsoft/qlib) is Microsoft's open-source quantitative investment platform. It standardises stock data storage (binary `.bin` files), provides feature engineering pipelines (Alpha158), and offers a full experiment workflow including signal evaluation and portfolio backtesting.

---

## How this project works

1. Loads local qlib stock data (OHLCV binary files at `~/Desktop/my_qlib_data`)
2. Computes **Alpha158** — 158 standard financial factors — via qlib's built-in handler
3. Trains the MASTER model 3 times (seeds 0, 1, 2) to predict each stock's 5-day forward return
4. For each seed, runs the full qlib workflow:
   - **SignalRecord** — generates predictions for the test period and saves them
   - **SigAnaRecord** — computes IC, ICIR, Rank IC, Rank ICIR
   - **PortAnaRecord** — runs a daily portfolio backtest (Top-30 stocks, daily rebalance)
5. Prints mean ± std of all metrics across the 3 seeds

---

## Repository files

| File | Role |
|------|------|
| `master.py` | **Author's exact standalone code** — MASTER neural network (`MASTER` nn.Module) and standalone `MASTERModel(SequenceModel)`. Not modified. Source: [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER) |
| `base_model.py` | **Author's exact standalone code** — `SequenceModel` base class (standalone training loop), `DailyBatchSamplerRandom`, and utility functions (`zscore`, `drop_extreme`, `calc_ic`, …). Not modified. Source: [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER) |
| `master_model.py` | **Integration bridge** — imports `MASTER` from `master.py`, wraps it in a qlib-compatible `MASTERModel(qlib.Model)` with `fit(dataset)`, `predict(dataset)`, `load_model(path)`. This is the glue between the two author repos. |
| `main.py` | **Entry point** — qlib workflow: initialises qlib, builds the Alpha158 dataset, trains 3 seeds, runs `SignalRecord → SigAnaRecord → PortAnaRecord`. Based on [SJTU-DMTai/qlib examples/benchmarks/MASTER](https://github.com/SJTU-DMTai/qlib/tree/fbd067c0f0a58a9ba0399c5334d588f7f30c9b10/examples/benchmarks/MASTER). |
| `workflow_config_master_Alpha158.yaml` | **Primary config** — qlib-style YAML with Alpha158 features, our local data path, date ranges, model hyperparameters, and portfolio backtest settings. |
| `workflow_config_master.yaml` | Legacy config using 10 custom OHLCV features. Kept for reference; not used by `main.py`. |
| `base_model.py` | Author's standalone utilities (unchanged). |
| `requirements.txt` | Python package dependencies. |

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

Your qlib data folder (default: `~/Desktop/my_qlib_data`) must follow this layout:

```
my_qlib_data/
├── calendars/
│   └── day.txt              # one trading date per line, e.g.: 2020-01-02
├── features/
│   └── sh600000/            # one sub-folder per stock
│       ├── close.day.bin
│       ├── open.day.bin
│       ├── high.day.bin
│       ├── low.day.bin
│       ├── volume.day.bin
│       ├── amount.day.bin
│       ├── vwap.day.bin
│       └── ...
└── instruments/
    ├── csi300.txt           # CSI 300 constituent list (with date ranges)
    ├── csi500.txt
    └── ...
```

**Instruments file format** — tab-separated, no header:
```
SH600000    2010-01-04    2026-03-20
SZ000001    2010-01-04    2026-03-20
```

**Alpha158 requirements**: the handler needs `$close`, `$open`, `$high`, `$low`, `$volume`, `$amount`, `$vwap` fields. Any standard qlib A-share dataset provides these.

---

## Quick start

**Step 1 — Check the config**

Open `workflow_config_master_Alpha158.yaml` and verify:
- `provider_uri`: path to your qlib data folder (default `~/Desktop/my_qlib_data`)
- `segments`: train / valid / test date splits

**Step 2 — Run**

```bash
# Train 3 seeds and run full backtest
python main.py

# Or specify a different config
python main.py --config workflow_config_master_Alpha158.yaml
```

**Step 3 — Backtest only** (skip training, load existing checkpoints)

```bash
python main.py --only_backtest
```

This loads `model/csi300master_0.pkl`, `model/csi300master_1.pkl`, `model/csi300master_2.pkl` and runs the qlib workflow (signal + portfolio analysis) without retraining.

---

## What `main.py` outputs

### During training (each epoch)
```
Epoch 5, train_loss 0.9987, valid ic 0.0318, icir 0.423, rankic 0.0291, rankicir 0.395.
```

| Column | Meaning |
|--------|---------|
| `train_loss` | MSE on training data (z-scored labels; starts near 1.0, normal) |
| `valid ic` | Mean daily IC on the validation set |
| `icir` | IC / std(IC) — consistency of the IC signal |
| `rankic` | Mean daily Rank IC (Spearman, more robust to outliers) |
| `rankicir` | Rank IC / std(Rank IC) |

### After training — qlib signal analysis (`SigAnaRecord`)

For each seed, the following metrics are printed and saved in the qlib experiment recorder:

| Metric | Meaning |
|--------|---------|
| `IC` | Mean daily Pearson correlation between model scores and actual returns (test period). Meaningful > 0.02, strong > 0.05 |
| `ICIR` | IC ÷ std(IC). Good > 0.3. Measures how consistent the signal is over time |
| `Rank IC` | Same as IC but Spearman (rank-based). More robust to outlier returns |
| `Rank ICIR` | Rank IC ÷ std(Rank IC) |

### After training — qlib portfolio backtest (`PortAnaRecord`)

The backtest uses `TopkDropoutStrategy`: every day, hold the **top 30 stocks** by model score, replacing stocks that drop out of the top 30.

Key backtest metrics:

| Metric | Meaning |
|--------|---------|
| `1day.excess_return_without_cost.annualized_return` | Annualised alpha return vs benchmark (CSI 300). Positive = outperformed the index |
| `1day.excess_return_without_cost.information_ratio` | Annualised IR of the daily excess return. > 1.0 is considered good |

> **Without transaction costs**: the backtest above ignores commissions and slippage. Real performance will be lower due to turnover costs.

### Summary across seeds

After all 3 seeds, a summary is printed:
```
Summary (mean ± std across 3 seeds):
  IC: 0.0341 ± 0.0023
  ICIR: 0.4512 ± 0.0187
  Rank IC: 0.0312 ± 0.0019
  Rank ICIR: 0.4221 ± 0.0201
  1day.excess_return_without_cost.annualized_return: 0.1523 ± 0.0312
  1day.excess_return_without_cost.information_ratio: 1.2341 ± 0.1823
```

---

## Model architecture

```
Input: (N stocks, T=8 days, 158 Alpha158 features)
    │
    ├─ [Optional Gate]       Market-guided feature selection (disabled — no market index data)
    ├─ Linear projection     158 → 256 (d_model)
    ├─ Positional Encoding   adds day-position signal (sinusoidal)
    ├─ Temporal Attention    each stock attends to its own past 8 days
    ├─ Spatial Attention     stocks attend to each other (cross-sectional)
    ├─ Temporal Aggregation  compress 8 time steps → 1 vector (weighted by last-day query)
    └─ Linear decoder        256 → 1 score per stock
```

**Gate note**: the original MASTER uses 63 market-index features (CSI 300/500/903 technical factors) to adaptively gate the 158 Alpha158 features. Since we use only standard qlib data without a separate market-index handler, the Gate is disabled (`gate_input_start_index == gate_input_end_index == 158`). The model still runs the full temporal + spatial attention pipeline.

---

## Features: Alpha158

Alpha158 is qlib's built-in set of **158 standard financial factors** automatically computed from raw OHLCV data. They cover:

- **Return features**: 5/10/20/30/60-day close/open/high/low/vwap returns and their ratios
- **Volatility features**: rolling standard deviations of returns at various windows
- **Volume features**: rolling mean/std of volume and amount ratios
- **Momentum features**: various cross-period return comparisons
- **Technical features**: RSI, Bollinger-band-like ratios, intraday range

All 158 features are normalised with `RobustZScoreNorm` (fitted on the training period only, applied consistently to validation and test to prevent data leakage).

**Label**: 5-day forward return — `Ref($close, -2) / Ref($close, -1) - 1`, cross-sectionally rank-normalised (`CSRankNorm`) during training.

---

## Config overview (`workflow_config_master_Alpha158.yaml`)

```yaml
qlib_init:
    provider_uri: ~/Desktop/my_qlib_data   # ← your local data path

market: csi300
benchmark: SH000300

data_handler_config:
    start_time: 2010-01-01
    end_time:   2026-03-20
    fit_start_time: 2010-01-01             # normalisation fitted on this window only
    fit_end_time:   2014-12-31
    learn_processors: [DropnaLabel, CSRankNorm]   # training labels
    infer_processors: [RobustZScoreNorm, Fillna]  # test features

task:
    model:
        class: MASTERModel
        module_path: master_model          # local bridge module
        kwargs:
            d_feat: 158                    # Alpha158
            gate_input_start_index: 158    # gate disabled (= d_feat)
            gate_input_end_index: 158
            n_epochs: 40
            lr: 0.000008
            train_stop_loss_thred: 0.95

    dataset:
        class: TSDatasetH
        kwargs:
            segments:
                train: [2010-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test:  [2017-01-01, 2026-03-20]
            step_len: 8                    # 8-day lookback window

port_analysis_config:
    strategy: TopkDropoutStrategy (top 30 stocks, daily rebalance)
    backtest: 2017-01-01 to 2026-03-20, benchmark SH000300
```

---

## qlib Experiment Records

The workflow stores all results using qlib's `R` (Recorder). Each seed creates an experiment named `workflow_seed{seed}` under `~/.qlib/mlruns/`. You can inspect results with:

```bash
# List all experiments
mlflow ui   # then open http://localhost:5000
```

Or access programmatically:
```python
from qlib.workflow import R
with R.start(experiment_name="workflow_seed0"):
    recorder = R.get_recorder()
    metrics = recorder.list_metrics()
    predictions = recorder.load_object("pred.pkl")
```

---

## Integration design

This project integrates two separate author repositories without modifying any author code:

| Source | Files used | Role |
|--------|-----------|------|
| [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER) | `master.py`, `base_model.py` | Neural network architecture + standalone training utilities |
| [SJTU-DMTai/qlib examples/benchmarks/MASTER](https://github.com/SJTU-DMTai/qlib/tree/fbd067c0f0a58a9ba0399c5334d588f7f30c9b10/examples/benchmarks/MASTER) | `main.py` structure, `workflow_config_master_Alpha158.yaml` | qlib workflow running approach |
| This repo | `master_model.py` | Bridge: imports `MASTER` from `master.py`, adds qlib `Model` interface |
