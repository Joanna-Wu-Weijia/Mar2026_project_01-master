# Integration bridge between the two author repositories:
#   - Neural network architecture : SJTU-DMTai/MASTER  (master.py / base_model.py)
#   - qlib workflow running approach: SJTU-DMTai/qlib   (main.py / workflow_config_master_Alpha158.yaml)
#
# This file imports the MASTER nn.Module from master.py (author's exact code, unchanged)
# and wraps it with a qlib-compatible Model interface so that the standard qlib workflow
# (R.start, SignalRecord, SigAnaRecord, PortAnaRecord) can use it directly.

import copy
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# ── Author's standalone code (unchanged) ─────────────────────────────────────
from master import MASTER                                     # neural network
from base_model import (                                      # shared utilities
    calc_ic,
    zscore,
    drop_extreme,
    drop_na,
    DailyBatchSamplerRandom,
)
# ─────────────────────────────────────────────────────────────────────────────


class MASTERModel(Model):
    """qlib-compatible wrapper around the MASTER neural network.

    Bridges:
      • SJTU-DMTai/MASTER  →  provides the MASTER nn.Module (master.py)
      • SJTU-DMTai/qlib    →  provides the running approach  (main.py + yaml)

    The qlib workflow calls:
        model.fit(dataset)          – train
        model.predict(dataset)      – inference  (used by SignalRecord)
        model.load_model(path)      – load checkpoint  (used by --only_backtest)
    """

    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        t_nhead: int = 4,
        s_nhead: int = 2,
        gate_input_start_index: int = 158,
        gate_input_end_index: int = 158,
        T_dropout_rate: float = 0.5,
        S_dropout_rate: float = 0.5,
        beta: float = 1.0,
        n_epochs: int = 40,
        lr: float = 8e-6,
        GPU: int = 0,
        seed: int = 0,
        train_stop_loss_thred: float = None,
        save_path: str = "model/",
        save_prefix: str = "",
        market: str = "csi300",
        benchmark: str = "SH000300",
    ):
        self.d_feat = d_feat
        self.d_model = d_model
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.beta = beta
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.market = market
        self.benchmark = benchmark
        self.fitted = False

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Instantiate the MASTER nn.Module from master.py (author's exact code)
        self.model = MASTER(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate,
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta,
        )
        self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def load_model(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask] - label[mask]) ** 2
        return torch.mean(loss)

    # ── Training / evaluation epochs ─────────────────────────────────────────

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label)  # CSZscoreNorm
            #########################

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # Note the difference:
            # 1) The qlib.DropnaLabel drop **samples** according to label.
            # 2) Here we use all samples to compute the inter-stock correlation, but only drop the na labels to compute metrics (loss, etc.).
            # 3) If you already used qlib.DropnaLabel to process the validation data, this will do nothing.
            mask, label = drop_na(label)
            label = zscore(label)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred[mask], label)
            losses.append(loss.item())

        return float(np.mean(losses))

    # ── Data loader ──────────────────────────────────────────────────────────

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    # ── qlib Model interface ─────────────────────────────────────────────────

    def fit(self, dataset: DatasetH):
        import os
        os.makedirs(self.save_path, exist_ok=True)

        # DK_L (learn processors) for training — e.g. DropnaLabel + CSRankNorm
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # DK_I (infer processors) for validation — normalisation fitted on training only
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=False)

        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)

            predictions, metrics = self._predict_metrics(valid_loader)
            print(
                "Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (
                    step, train_loss,
                    metrics["IC"], metrics["ICIR"],
                    metrics["RIC"], metrics["RICIR"],
                )
            )

            if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f"{self.save_path}/{self.save_prefix}master_{self.seed}.pkl")
                break

        self.fitted = True
        if best_param is None:
            best_param = self.model.state_dict()
        torch.save(best_param, f"{self.save_path}/{self.save_prefix}master_{self.seed}.pkl")
        print(f"Model saved to {self.save_path}/{self.save_prefix}master_{self.seed}.pkl")

    def predict(self, dataset: DatasetH, segment: str = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        # DK_I (infer processors) for test — same normalisation as validation
        dl_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _predict_metrics(self, data_loader):
        """Compute predictions + IC/RankIC metrics on a data_loader."""
        self.model.eval()
        preds, ic, ric = [], [], []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        predictions = pd.Series(
            np.concatenate(preds),
            index=data_loader.sampler.data_source.get_index(),
        )
        metrics = {
            "IC":    np.mean(ic),
            "ICIR":  np.mean(ic) / np.std(ic),
            "RIC":   np.mean(ric),
            "RICIR": np.mean(ric) / np.std(ric),
        }
        return predictions, metrics
