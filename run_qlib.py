"""
run_qlib.py — Train and evaluate the MASTER model using your local qlib data.

Usage:
    python run_qlib.py [--mode train|predict|both] [--instrument csi300] \
                       [--data_path ~/Desktop/my_qlib_data] [--seed 0]

Your qlib data folder should have the structure:
    my_qlib_data/
    ├── calendars/
    │   └── day.txt          (one trading date per line, e.g. 2010-01-04)
    ├── features/
    │   └── <symbol>/        (e.g. sh600000, sz000001, bj430090)
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
        ├── csi300.txt       (symbol  start_date  end_date, tab-separated)
        ├── csi500.txt
        └── ...

Instruments file format (tab-separated, no header):
    SH600000    2010-01-04    2023-12-29
    SZ000001    2010-01-04    2023-12-29
    ...
"""

import argparse
import os

import numpy as np
import pandas as pd

import qlib
from qlib.config import REG_CN
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

from master_model import MASTERModel


# ---------------------------------------------------------------------------
# Feature definitions — built from your available fields:
#   close, open, high, low, volume, amount, change, factor, vwap, adjclose
#
# 10 features total:
#   OPEN_RET    open-to-previous-close return
#   HIGH_RET    high-to-previous-close return
#   LOW_RET     low-to-previous-close return
#   CLOSE_RET   close return (daily return)
#   VWAP_RET    vwap-to-previous-close return
#   LOG_VOL     log(volume + 1)
#   LOG_AMT     log(amount + 1)
#   INTRADAY    (close - low) / (high - low)  — intraday price position
#   MOM5        5-day momentum (yesterday's close vs 5 days ago)
#   MOM10       10-day momentum (yesterday's close vs 10 days ago)
# ---------------------------------------------------------------------------
FEATURE_FIELDS = [
    "$open/Ref($close,1)-1",
    "$high/Ref($close,1)-1",
    "$low/Ref($close,1)-1",
    "$close/Ref($close,1)-1",
    "$vwap/Ref($close,1)-1",
    "Ln($volume+1)",
    "Ln($amount+1)",
    "($close-$low)/($high-$low+1e-8)",
    "Ref($close,1)/Ref($close,6)-1",
    "Ref($close,1)/Ref($close,11)-1",
]
FEATURE_NAMES = [
    "OPEN_RET", "HIGH_RET", "LOW_RET", "CLOSE_RET", "VWAP_RET",
    "LOG_VOL", "LOG_AMT", "INTRADAY", "MOM5", "MOM10",
]

# Label: next trading day's return
#   Ref($close, -2) / Ref($close, -1) - 1
#   i.e., tomorrow's close / today's close - 1
LABEL_FIELD = "Ref($close,-2)/Ref($close,-1)-1"
LABEL_NAME = "LABEL0"

D_FEAT = len(FEATURE_FIELDS)  # 10


def build_dataset(
    instruments: str,
    data_start: str,
    data_end: str,
    fit_start: str,
    fit_end: str,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    lookback: int = 8,
) -> DatasetH:
    """
    Build a qlib DatasetH with RobustZScoreNorm preprocessing.

    The key fix from qlib-update (pytorch_master_ts.py):
      - learn_processors  → used for training data  (DK_L)
      - infer_processors  → used for valid/test data (DK_I)
    Both processors fit their normalisation statistics on [fit_start, fit_end]
    (training period) and then apply them consistently.
    """
    handler_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "start_time": data_start,
            "end_time": data_end,
            "fit_start_time": fit_start,
            "fit_end_time": fit_end,
            "instruments": instruments,
            "data_loader": {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": (FEATURE_FIELDS, FEATURE_NAMES),
                        "label": ([LABEL_FIELD], [LABEL_NAME]),
                    },
                    "freq": "day",
                },
            },
            # learn_processors: applied when data_key=DK_L (training)
            "learn_processors": [
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {"fields_group": "feature", "clip_outlier": True},
                },
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            # infer_processors: applied when data_key=DK_I (valid / test)
            # Uses statistics fitted on the training period — this is the
            # bug-fix introduced in qlib-update.
            "infer_processors": [
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {"fields_group": "feature", "clip_outlier": True},
                },
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
        },
    }

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
            "step_len": lookback,
        },
    }

    dataset = init_instance_by_config(dataset_config)
    return dataset


def calc_metrics(predictions: pd.DataFrame, dataset: DatasetH) -> dict:
    """Compute IC, ICIR, RankIC, RankICIR from predictions vs true labels."""
    dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    label_df = dl_test.data_arr  # raw numpy, last column is label

    # Rebuild a Series from the test DataHandler
    handler = dataset.handler
    label_series = handler.fetch(
        selector=slice(None), level="datetime", col_set="label", data_key=DataHandlerLP.DK_I
    ).iloc[:, 0]

    pred_series = predictions.iloc[:, 0]
    merged = pd.concat([pred_series.rename("pred"), label_series.rename("label")], axis=1).dropna()

    ic_list, ric_list = [], []
    for date, group in merged.groupby(level="datetime"):
        if len(group) < 5:
            continue
        ic = group["pred"].corr(group["label"])
        ric = group["pred"].corr(group["label"], method="spearman")
        ic_list.append(ic)
        ric_list.append(ric)

    metrics = {
        "IC": np.nanmean(ic_list),
        "ICIR": np.nanmean(ic_list) / np.nanstd(ic_list),
        "RIC": np.nanmean(ric_list),
        "RICIR": np.nanmean(ric_list) / np.nanstd(ric_list),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="MASTER model with local qlib data")
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="both",
                        help="Run mode: train only, predict only (requires saved model), or both")
    parser.add_argument("--data_path", type=str, default="~/Desktop/my_qlib_data",
                        help="Path to your qlib data folder")
    parser.add_argument("--instrument", type=str, default="csi300",
                        help="Instrument universe, e.g. csi300, csi500, csi800, all")
    parser.add_argument("--data_start", type=str, default="2010-01-01",
                        help="Earliest date to load from qlib data")
    parser.add_argument("--data_end", type=str, default="2023-12-31",
                        help="Latest date to load from qlib data")
    parser.add_argument("--train_start", type=str, default="2010-01-01")
    parser.add_argument("--train_end", type=str, default="2019-12-31")
    parser.add_argument("--valid_start", type=str, default="2020-01-01")
    parser.add_argument("--valid_end", type=str, default="2020-12-31")
    parser.add_argument("--test_start", type=str, default="2021-01-01")
    parser.add_argument("--test_end", type=str, default="2023-12-31")
    parser.add_argument("--lookback", type=int, default=8,
                        help="Lookback window T (paper uses 8)")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--t_nhead", type=int, default=4)
    parser.add_argument("--s_nhead", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_stop_loss", type=float, default=None,
                        help="Stop training early when train_loss <= this threshold")
    parser.add_argument("--save_path", type=str, default="model/",
                        help="Directory for saving model checkpoints")
    parser.add_argument("--save_prefix", type=str, default="",
                        help="Prefix for checkpoint filenames")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pre-trained model .pkl file (for predict mode)")
    parser.add_argument("--output_csv", type=str, default="predictions.csv",
                        help="Where to save prediction output")
    parser.add_argument("--GPU", type=int, default=0,
                        help="GPU device index (ignored if no CUDA available)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Initialise qlib
    # ------------------------------------------------------------------
    data_path = os.path.expanduser(args.data_path)
    print(f"Initialising qlib with data path: {data_path}")
    qlib.init(provider_uri=data_path, region=REG_CN)

    # ------------------------------------------------------------------
    # 2. Build the dataset (DatasetH)
    # ------------------------------------------------------------------
    print(f"Building dataset for instrument={args.instrument}, "
          f"train={args.train_start}~{args.train_end}, "
          f"valid={args.valid_start}~{args.valid_end}, "
          f"test={args.test_start}~{args.test_end}, "
          f"lookback={args.lookback}")
    dataset = build_dataset(
        instruments=args.instrument,
        data_start=args.data_start,
        data_end=args.data_end,
        fit_start=args.train_start,
        fit_end=args.train_end,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
        lookback=args.lookback,
    )

    # ------------------------------------------------------------------
    # 3. Build the MASTER model
    #    gate_input_start_index == gate_input_end_index == D_FEAT (10)
    #    → market gate is disabled; the model runs in no-gate mode
    # ------------------------------------------------------------------
    model = MASTERModel(
        d_feat=D_FEAT,
        d_model=args.d_model,
        t_nhead=args.t_nhead,
        s_nhead=args.s_nhead,
        gate_input_start_index=D_FEAT,   # no market features → gate disabled
        gate_input_end_index=D_FEAT,
        T_dropout_rate=args.dropout,
        S_dropout_rate=args.dropout,
        beta=None,
        n_epochs=args.n_epochs,
        lr=args.lr,
        GPU=args.GPU,
        seed=args.seed,
        train_stop_loss_thred=args.train_stop_loss,
        save_path=args.save_path,
        save_prefix=args.save_prefix,
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    if args.mode in ("train", "both"):
        print("\n=== Training ===")
        model.fit(dataset)

    # ------------------------------------------------------------------
    # 5. Load pre-trained weights (for predict-only mode)
    # ------------------------------------------------------------------
    if args.mode == "predict":
        if args.model_path is None:
            # Try the default checkpoint path
            default_ckpt = f"{args.save_path}/{args.save_prefix}master_{args.seed}.pkl"
            if os.path.exists(default_ckpt):
                args.model_path = default_ckpt
            else:
                raise FileNotFoundError(
                    f"No model checkpoint found at {default_ckpt}. "
                    "Please train first (--mode train) or provide --model_path."
                )
        print(f"Loading model from {args.model_path}")
        model.load_model(args.model_path)

    # ------------------------------------------------------------------
    # 6. Predict on test set
    # ------------------------------------------------------------------
    if args.mode in ("predict", "both"):
        print("\n=== Predicting on test set ===")
        predictions = model.predict(dataset, segment="test")
        predictions.to_csv(args.output_csv)
        print(f"Predictions saved to {args.output_csv}")
        print(predictions.head(10))

        # Compute IC metrics using validation set for a quick sanity check
        print("\n=== Predicting on validation set ===")
        val_predictions = model.predict(dataset, segment="valid")

        # Quick daily IC from predictions vs labels
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_I)
        import torch
        from master_model import DailyBatchSamplerRandom
        from torch.utils.data import DataLoader

        sampler = DailyBatchSamplerRandom(dl_valid, shuffle=False)
        valid_loader = DataLoader(dl_valid, sampler=sampler, drop_last=False)

        ic_list, ric_list = [], []
        pred_idx = 0
        val_pred_arr = val_predictions.values.ravel()

        for data in valid_loader:
            data = torch.squeeze(data, dim=0)
            label = data[:, -1, -1].numpy()
            n = len(label)
            pred = val_pred_arr[pred_idx:pred_idx + n]
            pred_idx += n

            valid_mask = ~np.isnan(label)
            if valid_mask.sum() < 5:
                continue
            from scipy.stats import spearmanr
            ic = np.corrcoef(pred[valid_mask], label[valid_mask])[0, 1]
            ric, _ = spearmanr(pred[valid_mask], label[valid_mask])
            ic_list.append(ic)
            ric_list.append(ric)

        print(f"Valid IC:    {np.nanmean(ic_list):.4f}  ICIR: {np.nanmean(ic_list)/np.nanstd(ic_list):.4f}")
        print(f"Valid RankIC:{np.nanmean(ric_list):.4f}  RICIR:{np.nanmean(ric_list)/np.nanstd(ric_list):.4f}")


if __name__ == "__main__":
    main()
