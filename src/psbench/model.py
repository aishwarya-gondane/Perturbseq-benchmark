"""Predicting double-perturbation profiles from single perturbations.

Task: for held-out double perturbations A+B, predict the pseudo-bulk
expression change relative to control, given the other conditions.

Baselines
    additive      delta_A + delta_B
    control       no change (all zeros)
    mean_double   mean delta of the training doubles

Model
    A small PyTorch network. Each gene has a learned embedding (index 0 is a
    zero "no second gene" token, so singles and doubles share one input
    format). The pair representation is symmetric: [e_A + e_B, e_A * e_B].
    `mlp_direct` predicts the delta; `mlp_residual` predicts the departure
    from the additive baseline.

Evaluation is k-fold cross-validation over the double perturbations. All
single perturbations are always available for training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn

from .data import COMBO_SEP, CONTROL


@dataclass
class ConditionTable:
    singles: dict
    doubles: dict
    genes: list


def build_table(adata, control: str = CONTROL, min_cells: int = 10) -> ConditionTable:
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    ctrl = X[labels == control].mean(0)
    singles, doubles = {}, {}
    for p in np.unique(labels):
        if p == control:
            continue
        m = labels == p
        if m.sum() < min_cells:
            continue
        k = p.count(COMBO_SEP) + 1
        if k == 1:
            singles[p] = X[m].mean(0) - ctrl
        elif k == 2:
            doubles[p] = X[m].mean(0) - ctrl
    doubles = {d: v for d, v in doubles.items()
               if all(g in singles for g in d.split(COMBO_SEP))}
    return ConditionTable(singles, doubles, list(adata.var_names))


def additive(table: ConditionTable, cond: str) -> np.ndarray:
    return sum(table.singles[g] for g in cond.split(COMBO_SEP))


class PairMLP(nn.Module):
    def __init__(self, n_embed: int, n_genes: int, emb: int, hidden: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(n_embed + 1, emb, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(2 * emb, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_genes)
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        e = self.emb(idx)
        return self.net(torch.cat([e.sum(1), e[:, 0] * e[:, 1]], dim=1))


def fit_predict(table: ConditionTable, train_doubles, test_doubles, residual: bool,
                cfg: dict, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    gidx = {g: i + 1 for i, g in enumerate(sorted(table.singles))}
    delta = {**table.singles, **table.doubles}

    def ids(c):
        g = c.split(COMBO_SEP)
        return [gidx[x] for x in g] + [0] * (2 - len(g))

    def base(c):
        return additive(table, c)

    perm = [str(x) for x in rng.permutation(list(train_doubles))]
    n_val = int(round(0.1 * len(perm)))
    val, fit_d = perm[:n_val], perm[n_val:]
    fit_conds = list(table.singles) + fit_d

    scale = float(np.std(np.stack([delta[c] for c in fit_conds]))) or 1.0

    def tensors(conds):
        x = torch.tensor([ids(c) for c in conds], dtype=torch.long)
        y = np.stack([delta[c] - (base(c) if residual else 0.0) for c in conds]) / scale
        return x, torch.tensor(y, dtype=torch.float32)

    x_fit, y_fit = tensors(fit_conds)
    x_val, y_val = tensors(val) if val else (None, None)

    model = PairMLP(len(gidx), len(table.genes), cfg["emb"], cfg["hidden"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    best, best_state, wait = np.inf, None, 0
    for _ in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_fit), y_fit)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            monitor = loss_fn(model(x_val), y_val).item() if x_val is not None else loss.item()
        if monitor < best - 1e-6:
            best, wait = monitor, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)

    model.eval()
    test = [str(d) for d in test_doubles]
    with torch.no_grad():
        out = model(torch.tensor([ids(c) for c in test], dtype=torch.long)).numpy() * scale
    return {c: out[i] + (base(c) if residual else 0.0) for i, c in enumerate(test)}


def _pearson(p: np.ndarray, t: np.ndarray) -> float:
    if np.std(p) == 0 or np.std(t) == 0:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def score(pred: np.ndarray, truth: np.ndarray, top_k: int = 20) -> dict:
    top = np.argsort(np.abs(truth))[::-1][:top_k]
    return {
        "mse": float(np.mean((pred - truth) ** 2)),
        "pearson": _pearson(pred, truth),
        "mse_top": float(np.mean((pred[top] - truth[top]) ** 2)),
        "pearson_top": _pearson(pred[top], truth[top]),
    }


def run_cv(adata, cfg: dict, seed: int = 0) -> pd.DataFrame:
    table = build_table(adata, min_cells=cfg.get("min_cells", 10))
    doubles = sorted(table.doubles)
    k = cfg["folds"]
    if len(doubles) < k:
        raise ValueError(f"only {len(doubles)} usable double perturbations for {k} folds")

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(doubles), k)
    rows = []
    for f, test in enumerate(folds):
        test = [str(d) for d in test]
        train = [d for d in doubles if d not in set(test)]
        mean_double = np.mean([table.doubles[d] for d in train], axis=0)
        preds = {
            "additive": {d: additive(table, d) for d in test},
            "control": {d: np.zeros(len(table.genes)) for d in test},
            "mean_double": {d: mean_double for d in test},
            "mlp_direct": fit_predict(table, train, test, False, cfg, seed + f),
            "mlp_residual": fit_predict(table, train, test, True, cfg, seed + f),
        }
        for name, pred in preds.items():
            for d in test:
                rows.append({"fold": f, "condition": d, "model": name,
                             **score(pred[d], table.doubles[d])})
    return pd.DataFrame(rows)
