import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim


def calc_ic(pred, label):
    df = pd.DataFrame({"pred": pred, "label": label})
    ic = df["pred"].corr(df["label"])
    ric = df["pred"].corr(df["label"], method="spearman")
    return ic, ric


def zscore(x):
    """Cross-sectional z-score (population std + eps for stability)."""
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def drop_extreme(x):
    """Drop top/bottom 2.5% of label values by rank (training regularization)."""
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025 * N)
    # Use [k : N-k] — when k==0, indices[0:-0] is an empty slice in Python.
    filtered_indices = indices[percent_2_5 : N - percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]


def drop_na(x):
    mask = ~torch.isnan(x)
    return mask, x[mask]


def qlib_sample_index(data_source):
    """Index for aligning predictions: TSDataSampler.get_index() or DataFrame.index."""
    gi = getattr(data_source, "get_index", None)
    return gi() if callable(gi) else data_source.index


def sanitize_features(x: torch.Tensor) -> torch.Tensor:
    """Replace non-finite feature values so models do not propagate NaN through layers."""
    if not torch.is_floating_point(x):
        x = x.float()
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def mse_loss_finite(pred, label):
    """MSE on entries where both pred and label are finite."""
    mask = torch.isfinite(pred) & torch.isfinite(label)
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
    return ((pred[mask] - label[mask]) ** 2).mean()


class DailyBatchSamplerRandom(Sampler):
    """One batch = one calendar day (all instruments that day), optional shuffle of days."""

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.daily_count = (
            pd.Series(index=qlib_sample_index(self.data_source)).groupby("datetime").size().values
        )
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel:
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path="model/", save_prefix=""):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        return mse_loss_finite(pred, label)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = sanitize_features(data[:, :, 0:-1].to(self.device))
            label = data[:, -1, -1].to(self.device)

            mask, label = drop_extreme(label)
            if label.numel() < 2:
                continue
            feature = feature[mask, :, :]
            label = zscore(label)
            if not torch.all(torch.isfinite(label)):
                continue

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)
            if not torch.isfinite(loss):
                continue
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        if not losses:
            raise RuntimeError(
                "No training batches had valid labels after filtering; check data/labels."
            )
        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = sanitize_features(data[:, :, 0:-1].to(self.device))
            label = data[:, -1, -1].to(self.device)

            mask, label = drop_na(label)
            if label.numel() < 2:
                continue
            label = zscore(label)
            if not torch.all(torch.isfinite(label)):
                continue

            pred = self.model(feature)
            loss = self.loss_fn(pred[mask], label)
            if not torch.isfinite(loss):
                continue
            losses.append(loss.item())

        if not losses:
            return float("nan")
        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = "Previously trained."

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print(
                    "Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f."
                    % (step, train_loss, metrics["IC"], metrics["ICIR"], metrics["RIC"], metrics["RICIR"])
                )
            else:
                print("Epoch %d, train_loss %.6f" % (step, train_loss))

            if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f"{self.save_path}/{self.save_prefix}_{self.seed}.pkl")
                break

    def predict(self, dl_test):
        if self.fitted == -1:
            raise ValueError("model is not fitted yet!")
        else:
            print("Epoch:", self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = sanitize_features(data[:, :, 0:-1].to(self.device))
            label = data[:, -1, -1]

            with torch.no_grad():
                pred = self.model(feature).detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=qlib_sample_index(dl_test))

        metrics = {
            "IC": np.mean(ic),
            "ICIR": np.mean(ic) / np.std(ic),
            "RIC": np.mean(ric),
            "RICIR": np.mean(ric) / np.std(ric),
        }

        return predictions, metrics
