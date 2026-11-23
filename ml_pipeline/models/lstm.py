"""LSTM models for sequence regression and classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMModel:
    """Sklearn-like wrapper around a PyTorch LSTM."""

    def __init__(
        self,
        task: str = "regression",
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 64,
        patience: int = 4,
        device: str | None = None,
        n_classes: int | None = None,
        class_weight: str | np.ndarray | None = None,
        seed: int = 42,
    ):
        self.task = task
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.n_classes = n_classes
        self.class_weight = class_weight
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LSTMNet | None = None
        self.input_dim: int | None = None
        # Regression targets are standardized for training (raw RUL of 0-125
        # with MSE made the LSTM collapse to predicting the mean on hard sets).
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        self._set_seed()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.input_dim = X.shape[-1]

        if self.task == "classification":
            y = y.astype(np.int64)
            self.n_classes = int(self.n_classes or (int(y.max()) + 1))
            out_dim = self.n_classes
            weight = None
            if self.class_weight is not None:
                if isinstance(self.class_weight, str) and self.class_weight == "balanced":
                    counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
                    counts = np.maximum(counts, 1.0)
                    w = counts.sum() / (self.n_classes * counts)
                    weight = torch.tensor(w, dtype=torch.float32, device=self.device)
                else:
                    weight = torch.tensor(
                        np.asarray(self.class_weight, dtype=np.float32),
                        dtype=torch.float32,
                        device=self.device,
                    )
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            y = y.astype(np.float32).reshape(-1, 1)
            self.y_mean = float(y.mean())
            self.y_std = float(y.std()) or 1.0
            y = ((y - self.y_mean) / self.y_std).astype(np.float32)
            out_dim = 1
            loss_fn = nn.MSELoss()

        self.model = LSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=out_dim,
        ).to(self.device)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        gen = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, generator=gen)

        val_loader = None
        if X_val is not None and y_val is not None:
            Xv = np.asarray(X_val, dtype=np.float32)
            yv = np.asarray(y_val)
            if self.task == "classification":
                yv = yv.astype(np.int64)
            else:
                yv = yv.astype(np.float32).reshape(-1, 1)
                yv = ((yv - self.y_mean) / self.y_std).astype(np.float32)
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv)),
                batch_size=self.batch_size,
                shuffle=False,
            )

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_state = None
        best_val = float("inf")
        stale = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            if val_loader is None:
                continue

            self.model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)
                    total += float(loss_fn(pred, yb).item()) * len(xb)
                    n += len(xb)
            val_loss = total / max(n, 1)
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        X = np.asarray(X, dtype=np.float32)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.model.eval()
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                pred = self.model(xb.to(self.device))
                if self.task == "classification":
                    outs.append(pred.argmax(dim=1).cpu().numpy())
                else:
                    outs.append(pred.cpu().numpy().ravel() * self.y_std + self.y_mean)
        return np.concatenate(outs, axis=0)
