#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Based on SJTU-DMTai/qlib  examples/benchmarks/MASTER/main.py
(commit fbd067c0f0a58a9ba0399c5334d588f7f30c9b10)

Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is a script like this,
which nearly does the same thing as `qrun XXX.yaml`.

Adaptations for this repository
────────────────────────────────
• No auto-download: we use a local qlib data folder (~/Desktop/my_qlib_data).
• provider_uri and all other settings are read from the YAML config file.
• MASTERModel is imported from the local master_model.py bridge module.
• Standard TSDatasetH + Alpha158 is used (no custom SJTU-DMTai qlib fork needed).
"""

import sys
import os
from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.insert(0, str(DIRNAME))

import yaml
import argparse
import pprint as pp
import numpy as np

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord


def parse_args():
    parser = argparse.ArgumentParser(description="MASTER model – qlib workflow")
    parser.add_argument(
        "--config",
        type=str,
        default="workflow_config_master_Alpha158.yaml",
        help="Path to the YAML workflow config (default: workflow_config_master_Alpha158.yaml)",
    )
    parser.add_argument(
        "--only_backtest",
        action="store_true",
        help="Skip training and load pre-trained checkpoints for backtesting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Load YAML config ──────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ── Initialise qlib with local data ───────────────────────────────────────
    provider_uri = os.path.expanduser(config["qlib_init"]["provider_uri"])
    print(f"Initialising qlib  provider_uri={provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # ── Build / cache the data handler ────────────────────────────────────────
    # Preprocessing Alpha158 can be slow; we pickle the handler once and reuse it.
    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    seg = config["task"]["dataset"]["kwargs"]["segments"]
    train_start = seg["train"][0]
    test_end    = seg["test"][1]
    h_path = DIRNAME / f"handler_{train_start}_{test_end}.pkl"

    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        print(f"Preprocessed handler saved to {h_path}")

    # Point the dataset config at the cached handler
    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config["task"]["dataset"])

    # ── Ensure model save directory exists ────────────────────────────────────
    if not os.path.exists("./model"):
        os.mkdir("./model")

    # ── Metrics accumulator (3 seeds, same as original paper) ─────────────────
    all_metrics = {
        k: []
        for k in [
            "IC",
            "ICIR",
            "Rank IC",
            "Rank ICIR",
            "1day.excess_return_without_cost.annualized_return",
            "1day.excess_return_without_cost.information_ratio",
        ]
    }

    # ── Run 3 seeds ───────────────────────────────────────────────────────────
    for seed in range(0, 3):
        print("─" * 60)
        print(f"Seed: {seed}")

        config["task"]["model"]["kwargs"]["seed"] = seed
        model = init_instance_by_config(config["task"]["model"])

        # Train or load checkpoint
        if not args.only_backtest:
            model.fit(dataset=dataset)
        else:
            ckpt = f"./model/{config['market']}master_{seed}.pkl"
            print(f"Loading checkpoint: {ckpt}")
            model.load_model(ckpt)

        # ── qlib Workflow: Signal → Signal Analysis → Portfolio Analysis ──────
        with R.start(experiment_name=f"workflow_seed{seed}"):
            recorder = R.get_recorder()

            # Generate predictions and record them
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            # Signal analysis: IC, ICIR, Rank IC, Rank ICIR
            sar = SigAnaRecord(recorder)
            sar.generate()

            # Portfolio backtest with TopkDropoutStrategy
            par = PortAnaRecord(recorder, config["port_analysis_config"], "day")
            par.generate()

            metrics = recorder.list_metrics()
            print(metrics)
            for k in all_metrics.keys():
                all_metrics[k].append(metrics[k])
            pp.pprint(all_metrics)

    # ── Summary across seeds ──────────────────────────────────────────────────
    print("=" * 60)
    print("Summary (mean ± std across 3 seeds):")
    for k in all_metrics.keys():
        vals = all_metrics[k]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
