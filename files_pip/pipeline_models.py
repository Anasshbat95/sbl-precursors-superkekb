"""
pipeline_models.py - Steps 3 + 4 of the SBL pipeline: models and training strategy.

Training strategy (unsupervised, "learn what normal looks like"):
    * chronological split of the whole time span: train | calibration | test
    * "normal" rows = farther than normal_margin from EVERY SBL (so buffer + recovery + the
      risk window are never learned as normal)
    * scaling statistics and the model are fitted on TRAIN-normal rows only
    * the calibration period is used only to normalise scores and pick thresholds
    * SBL timestamps are never used as training labels

Models (both give one anomaly score per row, higher = more anomalous):
    "zscore"   baseline: max over features of |robust z-score| (median / IQR of train-normal)
    "iforest"  Isolation Forest on the robust-scaled features - no assumption of normally
               distributed data, works with few rows, quick to train

Scores of different blocks are fused on a common time grid AFTER scoring, so the raw blocks
never have to be merged.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from files.sbl_anomaly import dist_to_sbl


def make_split(data: dict, cfg: dict) -> dict:
    t0 = min(d.index.min() for d in data.values())
    t1 = max(d.index.max() for d in data.values())
    span = t1 - t0
    return dict(t0=t0, t1=t1,
                train_end=t0 + span * cfg["train_frac"],
                calib_end=t0 + span * (cfg["train_frac"] + cfg["calib_frac"]))


def normal_flags(index, sbl, cfg) -> np.ndarray:
    return dist_to_sbl(index, sbl) > pd.Timedelta(cfg["normal_margin"]).total_seconds()


def score_block(F: pd.DataFrame, sbl, cfg: dict, split: dict, method: str) -> pd.Series:
    """Anomaly score per row of one block's feature table (NaN where features are incomplete)."""
    F = F.dropna()
    train = normal_flags(F.index, sbl, cfg) & (F.index < split["train_end"])
    if train.sum() < cfg["min_train_rows"]:
        raise ValueError(f"Only {int(train.sum())} normal training rows (need {cfg['min_train_rows']}). "
                         "Lower normal_margin, raise train_frac, or check the BEAM_ON filter.")
    Xtr = F[train]

    med = Xtr.median()
    # robust scale = IQR/1.349 (equals the std for normal data, but ignores outliers), with a floor
    # of 10% of the std so near-constant columns cannot blow up into huge z-scores
    scale = np.maximum((Xtr.quantile(0.75) - Xtr.quantile(0.25)) / 1.349, 0.1 * Xtr.std())
    ok = scale.notna() & (scale > 0)
    Z = ((F.loc[:, ok] - med[ok]) / scale[ok]).clip(-50, 50)

    if method == "zscore":
        s = Z.abs().max(axis=1)
    elif method == "iforest":
        fit_rows = Z[train]
        if len(fit_rows) > cfg["max_train_rows"]:
            fit_rows = fit_rows.sample(cfg["max_train_rows"], random_state=cfg["random_state"])
        model = IsolationForest(n_estimators=cfg["n_estimators"], random_state=cfg["random_state"])
        model.fit(fit_rows.to_numpy())
        s = pd.Series(-model.score_samples(Z.to_numpy()), index=Z.index)
    else:
        raise ValueError(f"Unknown method '{method}' - use 'zscore' or 'iforest'.")
    return s.rename(method)


def fuse_scores(scores: dict, sbl, cfg: dict, split: dict, grid: str) -> pd.DataFrame:
    """Per-block scores -> common grid -> each block normalised so that 0 = median and 1 = the
    99th percentile of its calibration-normal scores. Returns one column per block; the fused
    score is simply .max(axis=1) of this table."""
    margin = pd.Timedelta(cfg["normal_margin"]).total_seconds()
    cols = {}
    for name, s in scores.items():
        g = s.resample(grid).max()
        normal = dist_to_sbl(g.index, sbl) > margin
        cal = g[normal & (g.index >= split["train_end"]) & (g.index < split["calib_end"])].dropna()
        if len(cal) < 20:
            print(f"  WARNING: block '{name}' has only {len(cal)} calibration points - left out of the fusion.")
            continue
        med, hi = cal.median(), cal.quantile(0.99)
        den = hi - med if hi - med > 0 else (cal.std() if cal.std() > 0 else 1.0)
        cols[name] = (g - med) / den
    if not cols:
        raise ValueError("No block has enough calibration data - widen the calibration period (calib_frac).")
    return pd.DataFrame(cols)
