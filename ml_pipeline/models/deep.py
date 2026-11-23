"""
Deep sequence models for the Continuous RUL Benchmark (CRB).

All networks map windows (B, T, F) -> RUL (B,) in standardized units and are
trained by one shared trainer (``DeepRUL``) so every architecture gets the same
optimizer, early stopping, target scaling and evaluation.

Baselines:  LSTM, BiLSTM-Attn, CNN1D, CNN-LSTM, TCN, Transformer
Proposed:   CATT  - Condition-Aware TCN-Transformer (architecture)
            TCT   - Trajectory-Consistent Training (method):
                      * consistency loss: predictions on two windows of the same
                        unit, d cycles apart, must differ like the true RUL
                        (by d, or 0 while capped)
                      * asymmetric loss: late predictions (pred > true) cost more,
                        matching the PHM score
                      * causal trajectory smoothing at inference, strength tuned
                        on validation only
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class CausalConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation)

    def forward(self, x):  # (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """Residual block of two dilated causal convolutions."""

    def __init__(self, cin: int, cout: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.c1 = CausalConv1d(cin, cout, kernel, dilation)
        self.c2 = CausalConv1d(cout, cout, kernel, dilation)
        self.n1 = nn.GroupNorm(1, cout)
        self.n2 = nn.GroupNorm(1, cout)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.drop(F.gelu(self.n1(self.c1(x))))
        h = self.drop(F.gelu(self.n2(self.c2(h))))
        return F.gelu(h + self.down(x))


class AttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h):  # (B, T, D) -> (B, D)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * h).sum(dim=1)


def _transformer(d_model: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
        batch_first=True, norm_first=True, activation="gelu",
    )
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


# --------------------------------------------------------------------------- #
# Baseline networks
# --------------------------------------------------------------------------- #


class LSTMNet(nn.Module):
    def __init__(self, n_features, seq_len, hidden=64, layers=2, dropout=0.2,
                 bidirectional=False, attention=False, **_):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        d = hidden * (2 if bidirectional else 1)
        self.pool = AttentionPool(d) if attention else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        z = self.pool(h) if self.pool is not None else h[:, -1]
        return self.head(z).squeeze(-1)


class CNN1DNet(nn.Module):
    """Deep 1D-CNN over time (in the spirit of Li et al., 2018)."""

    def __init__(self, n_features, seq_len, channels=32, kernel=9, layers=4, dropout=0.2, **_):
        super().__init__()
        blocks, c = [], n_features
        for _ in range(layers):
            blocks += [nn.Conv1d(c, channels, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(channels), nn.ReLU()]
            c = channels
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(channels * seq_len, 100), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(100, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)


class CNNLSTMNet(nn.Module):
    def __init__(self, n_features, seq_len, channels=64, kernel=5, hidden=64, dropout=0.2, **_):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, padding=kernel // 2), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2), nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.lstm = nn.LSTM(channels, hidden, 1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.head(h[:, -1]).squeeze(-1)


class TCNNet(nn.Module):
    def __init__(self, n_features, seq_len, channels=64, kernel=3, dilations=(1, 2, 4, 8), dropout=0.1, **_):
        super().__init__()
        blocks, c = [], n_features
        for d in dilations:
            blocks.append(TCNBlock(c, channels, kernel, d, dropout))
            c = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, x):
        return self.head(self.tcn(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, n_features, seq_len, d_model=64, heads=4, layers=2, dropout=0.1, **_):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.enc = _transformer(d_model, heads, layers, dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos[:, : x.size(1)])
        return self.head(h.mean(dim=1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Proposed architecture: CATT
# --------------------------------------------------------------------------- #


class CATTNet(nn.Module):
    """
    Condition-Aware TCN-Transformer.

    1. Condition-aware embedding: sensor channels are projected and modulated by
       FiLM (scale/shift) computed from the operating-condition channels, so
       the network learns regime-specific normalization (a learned version of
       per-regime clustering + scaling). FiLM starts as the identity.
    2. Multi-scale causal TCN: stacked dilated causal blocks; outputs of all
       scales are summed (skip aggregation) to capture short- and long-range
       degradation patterns.
    3. Transformer encoder over the multi-scale features for global context.
    4. Head on [last step ; attention-pooled summary].
    """

    def __init__(self, n_features, seq_len, cond_idx=(), d_model=64, kernel=3,
                 dilations=(1, 2, 4, 8), heads=4, layers=2, dropout=0.1, **_):
        super().__init__()
        cond = sorted({int(i) for i in cond_idx if 0 <= int(i) < n_features})
        sens = [i for i in range(n_features) if i not in cond] or list(range(n_features))
        self.register_buffer("cond_idx", torch.tensor(cond, dtype=torch.long), persistent=False)
        self.register_buffer("sens_idx", torch.tensor(sens, dtype=torch.long), persistent=False)
        self.inp = nn.Linear(len(sens), d_model)
        self.film = None
        if cond:
            self.film = nn.Sequential(nn.Linear(len(cond), d_model), nn.GELU(), nn.Linear(d_model, 2 * d_model))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        self.tcn = nn.ModuleList([TCNBlock(d_model, d_model, kernel, d, dropout) for d in dilations])
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.enc = _transformer(d_model, heads, layers, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.inp(x.index_select(-1, self.sens_idx))
        if self.film is not None:
            gamma, beta = self.film(x.index_select(-1, self.cond_idx)).chunk(2, dim=-1)
            h = h * (1.0 + gamma) + beta
        z = h.transpose(1, 2)
        skip = 0
        for blk in self.tcn:
            z = blk(z)
            skip = skip + z
        h = (skip / len(self.tcn)).transpose(1, 2)
        h = self.norm(self.enc(h + self.pos[:, : h.size(1)]))
        return self.head(torch.cat([h[:, -1], self.pool(h)], dim=-1)).squeeze(-1)


ARCHS = {
    "lstm": LSTMNet,
    "cnn1d": CNN1DNet,
    "cnn_lstm": CNNLSTMNet,
    "tcn": TCNNet,
    "transformer": TransformerNet,
    "catt": CATTNet,
}


# --------------------------------------------------------------------------- #
# Shared trainer
# --------------------------------------------------------------------------- #


def causal_smooth(pred: np.ndarray, units: np.ndarray, alpha: float, cap: float) -> np.ndarray:
    """
    Causal (online) smoothing along each unit's trajectory.
    prior_t = s_{t-1} - 1 (RUL drops one cycle per step; held while at the cap),
    s_t = alpha * pred_t + (1 - alpha) * prior_t. Uses only past predictions.
    """
    out = np.asarray(pred, dtype=float).copy()
    units = np.asarray(units)
    if len(out) == 0:
        return out
    starts = np.flatnonzero(np.r_[True, units[1:] != units[:-1]])
    ends = np.r_[starts[1:], len(out)]
    for a, b in zip(starts, ends):
        s = out[a]
        for t in range(a + 1, b):
            prior = s - 1.0 if s < cap - 0.5 else s
            s = alpha * out[t] + (1.0 - alpha) * prior
            out[t] = s
    return np.clip(out, 0.0, cap)


class DeepRUL:
    """
    Sklearn-like wrapper: fit(X, y, X_val, y_val, units=, val_units=) / predict(X).

    consistency  > 0 enables the TCT consistency loss (needs ``units``)
    late_penalty > 0 weights late predictions by (1 + late_penalty)
    smoothing    = "auto" tunes causal smoothing on validation; None disables it
    """

    uses_units = True

    def __init__(self, arch: str, arch_kwargs: dict | None = None, epochs: int = 40,
                 batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-4,
                 patience: int = 8, seed: int = 42, consistency: float = 0.0,
                 max_offset: int = 10, late_penalty: float = 0.0, smoothing: str | None = None,
                 device: str | None = None, verbose: bool = False):
        if arch not in ARCHS:
            raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHS)}")
        self.arch = arch
        self.arch_kwargs = dict(arch_kwargs or {})
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.consistency = consistency
        self.max_offset = max_offset
        self.late_penalty = late_penalty
        self.smoothing = smoothing
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.net: nn.Module | None = None
        self.y_mean, self.y_std, self.cap = 0.0, 1.0, float("inf")
        self.alpha_: float | None = None
        self.history_: list[dict] = []

    # -- helpers ---------------------------------------------------------- #
    def _seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _tensor(self, a, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=self.device)

    @torch.no_grad()
    def _forward_all(self, X_t: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        bs = max(self.batch_size * 8, 1024)
        return torch.cat([self.net(X_t[i : i + bs]) for i in range(0, len(X_t), bs)]) if len(X_t) else X_t.new_zeros(0)

    # -- API ---------------------------------------------------------------- #
    def fit(self, X, y, X_val=None, y_val=None, units=None, val_units=None):
        self._seed()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n, seq_len, n_feat = X.shape
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) or 1.0
        self.cap = float(y.max())

        self.net = ARCHS[self.arch](n_feat, seq_len, **self.arch_kwargs).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, self.patience // 3))

        X_t = self._tensor(X)
        y_t = self._tensor((y - self.y_mean) / self.y_std)
        u_t = None
        if self.consistency > 0 and units is not None:
            u_t = self._tensor(np.asarray(units).astype(np.int64), dtype=torch.long)
        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if has_val:
            Xv_t = self._tensor(np.asarray(X_val, dtype=np.float32))
            yv = np.asarray(y_val, dtype=np.float32).ravel()
            yv_t = self._tensor((yv - self.y_mean) / self.y_std)

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            tot, cnt = 0.0, 0
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                if len(idx) < 2:
                    continue
                pred = self.net(X_t[idx])
                err = pred - y_t[idx]
                w = 1.0 + self.late_penalty * (err > 0).float() if self.late_penalty > 0 else 1.0
                loss = (w * err.pow(2)).mean()
                if u_t is not None:
                    d = torch.randint(1, self.max_offset + 1, (len(idx),), generator=gen).to(self.device)
                    j = torch.clamp(idx + d, max=n - 1)
                    ok = u_t[j] == u_t[idx]
                    if ok.any():
                        i2, j2 = idx[ok], j[ok]
                        dp = self.net(X_t[j2]) - pred[ok]
                        dy = y_t[j2] - y_t[i2]
                        loss = loss + self.consistency * (dp - dy).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
                cnt += len(idx)

            if not has_val:
                continue
            val_mse = float((self._forward_all(Xv_t) - yv_t).pow(2).mean())
            sched.step(val_mse)
            self.history_.append({"epoch": epoch, "train_loss": tot / max(cnt, 1), "val_mse": val_mse})
            if self.verbose:
                print(f"    epoch {epoch:3d} train={tot / max(cnt, 1):.4f} val_rmse={math.sqrt(val_mse) * self.y_std:.3f}")
            if val_mse < best - 1e-6:
                best, stale = val_mse, 0
                best_state = copy.deepcopy({k: v.detach() for k, v in self.net.state_dict().items()})
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

        # Tune causal smoothing on validation only
        self.alpha_ = None
        if self.smoothing == "auto" and has_val and val_units is not None:
            pv = np.maximum(self.predict(X_val), 0.0)
            best_rmse = float(np.sqrt(np.mean((pv - yv) ** 2)))
            for a in (0.7, 0.5, 0.35, 0.25, 0.15, 0.1):
                ps = causal_smooth(pv, val_units, a, self.cap)
                r = float(np.sqrt(np.mean((ps - yv) ** 2)))
                if r < best_rmse - 1e-9:
                    best_rmse, self.alpha_ = r, a
        return self

    def predict(self, X):
        assert self.net is not None, "call fit() first"
        X_t = self._tensor(np.asarray(X, dtype=np.float32))
        return self._forward_all(X_t).float().cpu().numpy() * self.y_std + self.y_mean

    def postprocess(self, pred, units):
        """Causal trajectory smoothing (only if tuned on validation)."""
        if self.alpha_ is None or units is None:
            return pred
        return causal_smooth(pred, units, self.alpha_, self.cap)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters()) if self.net is not None else 0
