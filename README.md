# MASTER — Local Qlib Data Setup

This repository adapts the **MASTER** stock prediction model (AAAI-2024, [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER)) to run with your own local qlib-format data.

The starting point is `qlib-update/pytorch_master_ts.py` from the original repo, with minimal modifications described below.

---

## Files

| File | Purpose |
|------|---------|
| `workflow_config_master.yaml` | **Configuration file** — edit this to set your data path, dates, instrument universe, and model hyperparameters |
| `run_qlib.py` | **Entry point** — reads the config and runs training/evaluation. Pass `--config workflow_config_master.yaml` |
| `master_model.py` | MASTER neural net + `MASTERModel` class (adapted from `qlib-update/pytorch_master_ts.py`) |
| `base_model.py` | `SequenceModel` training framework (unchanged from original repo) |
| `requirements.txt` | Python dependencies |

`workflow_config_master.yaml` and `run_qlib.py` have different roles — the YAML is configuration, the Python script is execution. Edit the YAML, then run the script.

---

## What changed from the original `qlib-update/pytorch_master_ts.py`

1. **Relative → absolute imports** — the original uses `from ...data.dataset import DatasetH` (designed to live inside the qlib package). Changed to `from qlib.data.dataset import DatasetH` so the file works as a standalone script.

2. **No-gate mode** — the original MASTER uses 158 Alpha158 stock features + 63 market-index features; the market Gate weights stock features using market info. Since our data has no separate market-index features, the Gate is disabled when `gate_input_start_index == gate_input_end_index` (the default). The temporal + spatial attention layers are unchanged.

3. **Validation uses `DK_I`** — the key bug-fix from `qlib-update`: validation and test data now correctly use `infer_processors` (`DK_I`) instead of `learn_processors` (`DK_L`). Both processor sets fit their normalisation statistics on the training period only.

4. **`workflow_config_master.yaml`** — replaces `Alpha158` (158 factors) with a 10-feature `DataHandlerLP` built from your available binary fields, and replaces `MASTERTSDatasetH` (which needs market-index data) with the standard `DatasetH`.

---

## Data requirements

Your qlib data folder (`my_qlib_data`) must have the following structure:

```
my_qlib_data/
├── calendars/
│   └── day.txt              # one trading date per line: 2010-01-04
├── features/
│   └── <symbol>/            # e.g. sh600000, sz000001, bj430090
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
    ├── csi300.txt
    ├── csi500.txt
    ├── csi800.txt
    ├── csi1000.txt
    └── csiall.txt
```

**Instruments file format** (tab-separated, no header):
```
SH600000    2010-01-04    2023-12-29
SZ000001    2010-01-04    2023-12-29
BJ430090    2021-11-15    2023-12-29
```
The symbol prefix (`SH`/`SZ`/`BJ`) must match the folder name case used in `features/`.

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `pyqlib`, `torch`, `numpy`, `pandas`, `scipy`.

---

## Quick start

**Step 1**: Edit `workflow_config_master.yaml` — set your data path, dates, and instrument universe.

**Step 2**: Run:

```bash
python run_qlib.py --config workflow_config_master.yaml
```

You can override individual YAML values from the command line without editing the file:

```bash
python run_qlib.py \
    --config workflow_config_master.yaml \
    --instrument csi500 \
    --test_start 2022-01-01 \
    --test_end   2023-12-31 \
    --n_epochs   20
```

---

## Run modes

| Mode | Flag | What it does |
|------|------|--------------|
| Train + evaluate | `--mode both` | (default) trains the model and evaluates on valid + test |
| Train only | `--mode train` | trains and saves a checkpoint to `model/` |
| Predict only | `--mode predict` | loads a saved checkpoint and evaluates |

For predict-only mode, specify the checkpoint:
```bash
python run_qlib.py --config workflow_config_master.yaml \
    --mode predict \
    --model_path model/csi300master_0.pkl \
    --output_csv predictions.csv
```

---

## Features used

10 features derived from the available binary fields:

| Name | Expression | Description |
|------|-----------|-------------|
| `OPEN_RET` | `$open/Ref($close,1)-1` | open vs yesterday's close |
| `HIGH_RET` | `$high/Ref($close,1)-1` | high vs yesterday's close |
| `LOW_RET` | `$low/Ref($close,1)-1` | low vs yesterday's close |
| `CLOSE_RET` | `$close/Ref($close,1)-1` | daily return |
| `VWAP_RET` | `$vwap/Ref($close,1)-1` | VWAP vs yesterday's close |
| `LOG_VOL` | `Ln($volume+1)` | log volume |
| `LOG_AMT` | `Ln($amount+1)` | log traded amount |
| `INTRADAY` | `($close-$low)/($high-$low)` | intraday price position |
| `MOM5` | `Ref($close,1)/Ref($close,6)-1` | 5-day momentum |
| `MOM10` | `Ref($close,1)/Ref($close,11)-1` | 10-day momentum |

**Label**: `Ref($close,-2)/Ref($close,-1)-1` — next trading day's return.

**Lookback window T = 8** (same as the original paper). Each sample fed to the model is `(N, 8, 10)` where N is the number of stocks trading on that day.

---

## Model architecture

The MASTER architecture is unchanged from the original paper:

```
Input (N, T, d_feat)
    │
    ├─ [Gate]           weighted feature selection via market info (disabled here)
    │
    ├─ Linear           d_feat → d_model
    ├─ PositionalEncoding
    ├─ TAttention       intra-stock temporal attention
    ├─ SAttention       inter-stock spatial attention
    ├─ TemporalAttention  aggregate T steps → single vector
    └─ Linear           d_model → 1  (predicted return score)
```

Key hyperparameters (configurable in the YAML or via CLI):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_feat` | 10 | number of input features (auto-set from feature list) |
| `d_model` | 256 | transformer hidden dimension |
| `t_nhead` | 4 | temporal attention heads |
| `s_nhead` | 2 | spatial attention heads |
| `T_dropout_rate` | 0.5 | temporal attention dropout |
| `S_dropout_rate` | 0.5 | spatial attention dropout |
| `n_epochs` | 40 | maximum training epochs |
| `lr` | 1e-5 | Adam learning rate |
| `train_stop_loss_thred` | 0.95 | stop training when train loss ≤ this |

---

## Output

- **Model checkpoint**: `model/<save_prefix>master_<seed>.pkl`
- **Predictions CSV**: `predictions.csv` (or `--output_csv <path>`)
  - Index: `(datetime, instrument)` MultiIndex
  - Column: predicted return score (higher = more bullish)
- **Console metrics**: IC, ICIR, RankIC, RankICIR on validation and test sets
