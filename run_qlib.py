"""
run_qlib.py — Train and evaluate the MASTER model using your local qlib data.

Two ways to configure:
  1. Pass a YAML config file (recommended — mirrors the qlib workflow):
       python run_qlib.py --config workflow_config_master.yaml

  2. Pass CLI arguments directly (overrides YAML values when both given):
       python run_qlib.py --data_path ~/Desktop/my_qlib_data --instrument csi300

For the full qlib workflow (with experiment tracking and portfolio back-test),
you can also use qrun directly:
    PYTHONPATH=. qrun workflow_config_master.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

import qlib
from qlib.config import REG_CN
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

from master_model import MASTERModel, DailyBatchSamplerRandom


# ---------------------------------------------------------------------------
# Feature definitions — built from the 10 available binary fields:
#   close, open, high, low, volume, amount, change, factor, vwap, adjclose
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_FIELDS = [
    "$open/Ref($close,1)-1",
    "$high/Ref($close,1)-1",
    "$low/Ref($close,1)-1",
    "$close/Ref($close,1)-1",
    "$vwap/Ref($close,1)-1",
    "Log($volume+1)",
    "Log($amount+1)",
    "($close-$low)/($high-$low+1e-8)",
    "Ref($close,1)/Ref($close,6)-1",
    "Ref($close,1)/Ref($close,11)-1",
]
DEFAULT_FEATURE_NAMES = [
    "OPEN_RET", "HIGH_RET", "LOW_RET", "CLOSE_RET", "VWAP_RET",
    "LOG_VOL", "LOG_AMT", "INTRADAY", "MOM5", "MOM10",
]
DEFAULT_LABEL_FIELD = "Ref($close,-2)/Ref($close,-1)-1"
DEFAULT_LABEL_NAME  = "LABEL0"


def load_yaml_config(path: str) -> dict:
    """Load and resolve YAML anchors from a workflow config file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def extract_feature_lists(data_loader_cfg: dict):
    """
    Extract feature expressions and names from a data_loader config dict.
    Supports both inline lists and pre-resolved dicts from yaml.safe_load.
    Returns (fields: list[str], names: list[str]).
    """
    try:
        feature_cfg = data_loader_cfg["kwargs"]["config"]["feature"]
        # yaml.safe_load resolves the nested lists directly
        fields = feature_cfg[0]
        names  = feature_cfg[1]
        return fields, names
    except (KeyError, IndexError, TypeError):
        return DEFAULT_FEATURE_FIELDS, DEFAULT_FEATURE_NAMES


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
    feature_fields=None,
    feature_names=None,
    label_field: str = DEFAULT_LABEL_FIELD,
    label_name: str  = DEFAULT_LABEL_NAME,
    lookback: int = 8,
) -> TSDatasetH:
    """
    Build a qlib TSDatasetH (time-series windows + TSDataSampler) with RobustZScoreNorm.

    The key fix from qlib-update (pytorch_master_ts.py):
      - learn_processors  → used for training data  (DK_L)
      - infer_processors  → used for valid/test data (DK_I)
    Both fit their normalisation statistics on [fit_start, fit_end].
    """
    if feature_fields is None:
        feature_fields = DEFAULT_FEATURE_FIELDS
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    handler_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "start_time": data_start,
            "end_time": data_end,
            # fit_* belong on RobustZScoreNorm only; DataHandler rejects them.
            "instruments": instruments,
            "data_loader": {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": (feature_fields, feature_names),
                        "label":   ([label_field],   [label_name]),
                    },
                    "freq": "day",
                },
            },
            "learn_processors": [
                {"class": "RobustZScoreNorm",
                 "kwargs": {"fit_start_time": fit_start, "fit_end_time": fit_end,
                            "fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "infer_processors": [
                {"class": "RobustZScoreNorm",
                 "kwargs": {"fit_start_time": fit_start, "fit_end_time": fit_end,
                            "fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
        },
    }

    dataset_config = {
        "class": "TSDatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test":  (test_start,  test_end),
            },
            "step_len": lookback,
        },
    }

    return init_instance_by_config(dataset_config)


def eval_predictions(model: MASTERModel, dataset: DatasetH, segment: str = "valid"):
    """Run model.predict() and compute daily IC/RankIC metrics."""
    from torch.utils.data import DataLoader
    import torch
    from scipy.stats import spearmanr

    predictions = model.predict(dataset, segment=segment)

    dl = dataset.prepare(segment, col_set=["feature", "label"],
                         data_key=DataHandlerLP.DK_I)
    sampler = DailyBatchSamplerRandom(dl, shuffle=False)
    loader  = DataLoader(dl, sampler=sampler, drop_last=False)

    ic_list, ric_list = [], []
    pred_arr = predictions.values.ravel()
    pred_idx = 0

    for data in loader:
        data  = torch.squeeze(data, dim=0)
        label = data[:, -1, -1].numpy()
        n     = len(label)
        pred  = pred_arr[pred_idx:pred_idx + n]
        pred_idx += n

        mask = ~np.isnan(label)
        if mask.sum() < 5:
            continue
        ic  = np.corrcoef(pred[mask], label[mask])[0, 1]
        ric, _ = spearmanr(pred[mask], label[mask])
        ic_list.append(ic)
        ric_list.append(ric)

    metrics = {
        "IC":    np.nanmean(ic_list),
        "ICIR":  np.nanmean(ic_list) / np.nanstd(ic_list),
        "RIC":   np.nanmean(ric_list),
        "RICIR": np.nanmean(ric_list) / np.nanstd(ric_list),
    }
    return predictions, metrics


def main():
    parser = argparse.ArgumentParser(description="MASTER model with local qlib data")

    # YAML config (highest-level config source)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML workflow config (e.g. workflow_config_master.yaml). "
                             "CLI args below will override individual values in the YAML.")

    # Run mode
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="both")

    # Qlib init
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to your qlib data folder (default: from YAML or ~/Desktop/my_qlib_data)")

    # Data / instrument
    parser.add_argument("--instrument",   type=str,  default=None)
    parser.add_argument("--data_start",   type=str,  default=None)
    parser.add_argument("--data_end",     type=str,  default=None)
    parser.add_argument("--train_start",  type=str,  default=None)
    parser.add_argument("--train_end",    type=str,  default=None)
    parser.add_argument("--valid_start",  type=str,  default=None)
    parser.add_argument("--valid_end",    type=str,  default=None)
    parser.add_argument("--test_start",   type=str,  default=None)
    parser.add_argument("--test_end",     type=str,  default=None)
    parser.add_argument("--lookback",     type=int,  default=None,
                        help="Lookback window T (paper default: 8)")

    # Model hyper-parameters
    parser.add_argument("--d_model",      type=int,   default=None)
    parser.add_argument("--t_nhead",      type=int,   default=None)
    parser.add_argument("--s_nhead",      type=int,   default=None)
    parser.add_argument("--dropout",      type=float, default=None)
    parser.add_argument("--n_epochs",     type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--train_stop_loss", type=float, default=None)
    parser.add_argument("--save_path",    type=str,   default=None)
    parser.add_argument("--save_prefix",  type=str,   default=None)
    parser.add_argument("--GPU",          type=int,   default=None)

    # Predict-only
    parser.add_argument("--model_path",  type=str, default=None,
                        help="Pre-trained model .pkl (predict mode only)")
    parser.add_argument("--output_csv",  type=str, default="predictions.csv")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Merge: YAML defaults → CLI overrides
    # ------------------------------------------------------------------
    yaml_cfg = {}
    if args.config:
        yaml_cfg = load_yaml_config(args.config)

    def _get(cli_val, yaml_keys, fallback):
        """Return cli_val if set, else walk yaml_keys path, else fallback."""
        if cli_val is not None:
            return cli_val
        if not yaml_keys:
            # Empty path must not return the whole yaml_cfg (used for CLI-only segment dates).
            return fallback
        node = yaml_cfg
        for k in yaml_keys:
            if not isinstance(node, dict) or k not in node:
                return fallback
            node = node[k]
        return node if node is not None else fallback

    # qlib init
    data_path  = _get(args.data_path,  ["qlib_init", "provider_uri"], "~/Desktop/my_qlib_data")
    instrument = _get(args.instrument, ["market"],                     "csi300")

    # dates from data_handler_config
    dh = yaml_cfg.get("data_handler_config", {})
    data_start  = _get(args.data_start,  ["data_handler_config", "start_time"],      "2010-01-01")
    data_end    = _get(args.data_end,    ["data_handler_config", "end_time"],         "2023-12-31")
    fit_start   = _get(None,             ["fit_start"],   "2010-01-01")
    fit_end     = _get(None,             ["fit_end"],     "2019-12-31")

    # segment dates from task.dataset.kwargs.segments
    seg = yaml_cfg.get("task", {}).get("dataset", {}).get("kwargs", {}).get("segments", {})
    train_start = _get(args.train_start, [], None) or (seg.get("train") or ["2010-01-01"])[0]
    train_end   = _get(args.train_end,   [], None) or (seg.get("train") or [None, "2019-12-31"])[1]
    valid_start = _get(args.valid_start, [], None) or (seg.get("valid") or ["2020-01-01"])[0]
    valid_end   = _get(args.valid_end,   [], None) or (seg.get("valid") or [None, "2020-12-31"])[1]
    test_start  = _get(args.test_start,  [], None) or (seg.get("test")  or ["2021-01-01"])[0]
    test_end    = _get(args.test_end,    [], None) or (seg.get("test")  or [None, "2023-12-31"])[1]
    lookback    = _get(args.lookback,    ["task", "dataset", "kwargs", "step_len"], 8)

    # model kwargs
    mk = yaml_cfg.get("task", {}).get("model", {}).get("kwargs", {})
    d_feat       = mk.get("d_feat",       10)
    d_model      = _get(args.d_model,     ["task", "model", "kwargs", "d_model"],      256)
    t_nhead      = _get(args.t_nhead,     ["task", "model", "kwargs", "t_nhead"],      4)
    s_nhead      = _get(args.s_nhead,     ["task", "model", "kwargs", "s_nhead"],      2)
    dropout_t    = _get(args.dropout,     ["task", "model", "kwargs", "T_dropout_rate"], 0.5)
    dropout_s    = _get(args.dropout,     ["task", "model", "kwargs", "S_dropout_rate"], 0.5)
    n_epochs     = _get(args.n_epochs,    ["task", "model", "kwargs", "n_epochs"],     40)
    lr           = _get(args.lr,          ["task", "model", "kwargs", "lr"],           8e-6)
    seed         = _get(args.seed,        ["task", "model", "kwargs", "seed"],         0)
    stop_loss    = _get(args.train_stop_loss, ["task", "model", "kwargs", "train_stop_loss_thred"], None)
    save_path    = _get(args.save_path,   ["task", "model", "kwargs", "save_path"],    "model/")
    save_prefix  = _get(args.save_prefix, ["task", "model", "kwargs", "save_prefix"],  "")
    GPU          = _get(args.GPU,         ["task", "model", "kwargs", "GPU"],           0)
    gate_start   = mk.get("gate_input_start_index", d_feat)
    gate_end     = mk.get("gate_input_end_index",   d_feat)

    # features from YAML data_loader config (falls back to defaults)
    data_loader_cfg = yaml_cfg.get("data_handler_config", {}).get("data_loader", {})
    feature_fields, feature_names = extract_feature_lists(data_loader_cfg)
    d_feat = len(feature_fields)      # recompute from actual feature list

    # ------------------------------------------------------------------
    # 1. Initialise qlib
    # ------------------------------------------------------------------
    data_path = os.path.expanduser(str(data_path))
    print(f"Initialising qlib  data_path={data_path}")
    qlib.init(provider_uri=data_path, region=REG_CN)

    # ------------------------------------------------------------------
    # 2. Build TSDatasetH (time-series windows; prepare() → TSDataSampler)
    # ------------------------------------------------------------------
    print(f"Building dataset   instrument={instrument}  "
          f"train={train_start}~{train_end}  "
          f"valid={valid_start}~{valid_end}  "
          f"test={test_start}~{test_end}  "
          f"lookback={lookback}")
    dataset = build_dataset(
        instruments=instrument,
        data_start=str(data_start),
        data_end=str(data_end),
        fit_start=str(fit_start),
        fit_end=str(fit_end),
        train_start=str(train_start),
        train_end=str(train_end),
        valid_start=str(valid_start),
        valid_end=str(valid_end),
        test_start=str(test_start),
        test_end=str(test_end),
        feature_fields=feature_fields,
        feature_names=feature_names,
        lookback=int(lookback),
    )

    # ------------------------------------------------------------------
    # 3. Build the MASTER model
    # ------------------------------------------------------------------
    print(f"Building model     d_feat={d_feat}  d_model={d_model}  "
          f"gate={'enabled' if gate_end > gate_start else 'disabled (no market features)'}")
    model = MASTERModel(
        d_feat=d_feat,
        d_model=int(d_model),
        t_nhead=int(t_nhead),
        s_nhead=int(s_nhead),
        gate_input_start_index=int(gate_start),
        gate_input_end_index=int(gate_end),
        T_dropout_rate=float(dropout_t),
        S_dropout_rate=float(dropout_s),
        beta=None,
        n_epochs=int(n_epochs),
        lr=float(lr),
        GPU=int(GPU),
        seed=int(seed),
        train_stop_loss_thred=float(stop_loss) if stop_loss is not None else None,
        save_path=str(save_path),
        save_prefix=str(save_prefix),
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    if args.mode in ("train", "both"):
        print("\n=== Training ===")
        model.fit(dataset)

    # ------------------------------------------------------------------
    # 5. Load pre-trained weights (predict-only mode)
    # ------------------------------------------------------------------
    if args.mode == "predict":
        ckpt = args.model_path
        if ckpt is None:
            ckpt = f"{save_path}/{save_prefix}master_{seed}.pkl"
        ckpt = os.path.expanduser(ckpt)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"No checkpoint at {ckpt}. Train first (--mode train) or provide --model_path."
            )
        print(f"Loading model from {ckpt}")
        model.load_model(ckpt)

    # ------------------------------------------------------------------
    # 6. Evaluate on validation + test sets
    # ------------------------------------------------------------------
    if args.mode in ("predict", "both"):
        print("\n=== Validation set metrics ===")
        _, val_metrics = eval_predictions(model, dataset, segment="valid")
        print(f"  IC   : {val_metrics['IC']:.4f}    ICIR  : {val_metrics['ICIR']:.4f}")
        print(f"  RankIC: {val_metrics['RIC']:.4f}   RankICIR: {val_metrics['RICIR']:.4f}")

        print("\n=== Test set predictions ===")
        predictions, test_metrics = eval_predictions(model, dataset, segment="test")
        predictions.to_csv(args.output_csv)
        print(f"  IC   : {test_metrics['IC']:.4f}    ICIR  : {test_metrics['ICIR']:.4f}")
        print(f"  RankIC: {test_metrics['RIC']:.4f}   RankICIR: {test_metrics['RICIR']:.4f}")
        print(f"\nPredictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
