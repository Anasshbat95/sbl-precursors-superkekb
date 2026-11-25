"""
feature_scores.py - Anomaly score for EACH FEATURE ALONE, and heatmaps of the last N rows
before every SBL.

Idea:
    The pipeline gives one score per row (all features together). Here every feature gets its
    own 1D model, fitted only on train-normal rows (same rule as pipeline_models.score_block).
    Each feature score is then normalised on its calibration-normal rows:
        0 = median of normal operation, 1 = 99th percentile of normal operation
    so all features share one colour scale.

Methods (see extra_models.fit_scorer):
    "zscore"    |robust z|  (median / IQR of train-normal), NOT clipped
    "iforest", "lof", "ocsvm", "sgd_ocsvm", "mcd"   the same models, fitted on one column
    For one feature, "mcd" is just a robust (x - mu)^2 / sigma^2, i.e. close to z^2.

Usage (one block at a time - blocks are never merged):
    from feature_scores import per_feature_scores, plot_event_heatmaps, plot_event_overview
    S = per_feature_scores(features["background"], sbl, cfg, split, method="zscore")
    plot_event_heatmaps(S, sbl, split, n_points=10, top_k=20, block_name="background")
    plot_event_overview(S, sbl, split, n_points=10, top_k=30, block_name="background")
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extra_models import fit_scorer
from sbl_anomaly import dist_to_sbl


def _normal(index, sbl, cfg):
    return dist_to_sbl(index, sbl) > pd.Timedelta(cfg["normal_margin"]).total_seconds()


def per_feature_scores(F: pd.DataFrame, sbl, cfg: dict, split: dict, method="zscore",
                       max_fit_rows=5000) -> pd.DataFrame:
    """One normalised anomaly score per feature and row (same shape as F without NaN rows)."""
    F = F.dropna()
    normal = _normal(F.index, sbl, cfg)
    train = normal & (F.index < split["train_end"])
    calib = normal & (F.index >= split["train_end"]) & (F.index < split["calib_end"])
    if train.sum() < cfg["min_train_rows"] or calib.sum() < 20:
        raise ValueError(f"Too few normal rows (train={int(train.sum())}, calibration={int(calib.sum())}).")

    Xtr = F[train]
    med = Xtr.median()
    scale = np.maximum((Xtr.quantile(0.75) - Xtr.quantile(0.25)) / 1.349, 0.1 * Xtr.std())
    ok = scale.notna() & (scale > 0)
    Z = (F.loc[:, ok] - med[ok]) / scale[ok]

    rng = np.random.default_rng(cfg.get("random_state", 0))
    tr_idx = np.flatnonzero(train)
    if len(tr_idx) > max_fit_rows:                       # 1D models do not need all rows
        tr_idx = rng.choice(tr_idx, max_fit_rows, replace=False)

    out = {}
    for c in Z.columns:
        if method == "zscore":
            s = Z[c].abs().to_numpy()
        else:
            f = fit_scorer(method, Z[c].to_numpy()[tr_idx, None], cfg.get("random_state", 0))
            s = f(Z[c].to_numpy()[:, None])
        cal = s[calib]
        m, hi = np.median(cal), np.quantile(cal, 0.99)
        den = hi - m if hi - m > 0 else (cal.std() if cal.std() > 0 else 1.0)
        out[c] = (s - m) / den
    return pd.DataFrame(out, index=F.index)


def _events(S, sbl, split, period):
    sbl = pd.DatetimeIndex(sbl)
    if period == "test":
        sbl = sbl[sbl >= split["calib_end"]]
    return [t for t in sbl if S.index.min() < t <= S.index.max()]


def last_points(S: pd.DataFrame, t, n_points=10, buffer="0s") -> pd.DataFrame:
    """The last n rows strictly before t - buffer. Index = seconds before the SBL (negative)."""
    w = S[S.index < t - pd.Timedelta(buffer)].tail(n_points)
    w.index = (w.index - t).total_seconds()
    return w


def _top_features(S, events, n_points, buffer, top_k):
    pooled = pd.concat([last_points(S, t, n_points, buffer) for t in events])
    return pooled.max().sort_values(ascending=False).index[:top_k]


def plot_event_heatmaps(S, sbl, split, n_points=10, buffer="0s", top_k=20, period="test",
                        max_events=12, n_cols=4, vmax=3.0, block_name="block"):
    """One heatmap per SBL: rows = features, columns = the last n rows before the SBL.
    Colour = feature score (1 = 99th percentile of normal). Features: the top_k with the
    highest score over the shown events, same order in every panel."""
    events = _events(S, sbl, split, period)[:max_events]
    if not events:
        raise ValueError("No SBL with data in this block for the chosen period.")
    feats = _top_features(S, events, n_points, buffer, top_k)
    n_rows = int(np.ceil(len(events) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 0.22 * len(feats) * n_rows + 1.2),
                             squeeze=False, layout="constrained", sharey=True)
    im = None
    for ax, t in zip(axes.ravel(), events):
        w = last_points(S, t, n_points, buffer)[feats]
        im = ax.imshow(w.T.clip(lower=0).to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(w)))
        ax.set_xticklabels([f"{x:.0f}" for x in w.index], rotation=90, fontsize=7)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=7)
        ax.set_title(f"SBL {t:%Y-%m-%d %H:%M:%S}", fontsize=8)
        if len(w) < n_points:
            ax.text(0.5, -0.25, f"only {len(w)} rows", transform=ax.transAxes, ha="center",
                    fontsize=7, color="gray")
    for ax in axes.ravel()[len(events):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("seconds before SBL", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.6, label=f"feature score (1 = 99th pct. of normal, capped at {vmax:g})")
    fig.suptitle(f"{block_name}: per-feature anomaly score, last {n_points} rows before each SBL")
    return fig


def plot_event_overview(S, sbl, split, n_points=10, buffer="0s", top_k=30, period="test",
                        vmax=3.0, block_name="block"):
    """One heatmap for ALL events: rows = SBL events, columns = features,
    colour = maximum feature score over the last n rows before that SBL.
    Returns (figure, table)."""
    events = _events(S, sbl, split, period)
    if not events:
        raise ValueError("No SBL with data in this block for the chosen period.")
    feats = _top_features(S, events, n_points, buffer, top_k)
    tab = pd.DataFrame({t: last_points(S, t, n_points, buffer)[feats].max() for t in events}).T
    fig, ax = plt.subplots(figsize=(0.28 * len(feats) + 3, 0.22 * len(events) + 2), layout="constrained")
    im = ax.imshow(tab.clip(lower=0).to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([f"{t:%Y-%m-%d %H:%M}" for t in events], fontsize=7)
    ax.set_xlabel("feature")
    ax.set_ylabel("SBL event")
    fig.colorbar(im, ax=ax, shrink=0.7, label=f"max score, last {n_points} rows\n(1 = 99th pct. of normal)")
    ax.set_title(f"{block_name}: which feature is unusual right before which SBL?")
    return fig, tab
