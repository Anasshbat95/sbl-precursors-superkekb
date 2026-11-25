"""
pipeline_features.py - Step 2 of the SBL pipeline: feature engineering per block.

All features look only BACKWARD in time (no future data -> no leakage), and the window is
scaled to the block's own sampling rate (a 10 min window is useless if rows are 30 min apart).

Per original column c (all computed over the backward window):
    c__val     the current value
    c__mean    rolling mean
    c__std     rolling standard deviation
    c__range   rolling max - min
    c__slope   (value now - value one window ago) / window length   [change per second]
Rows without a full window of history (series start, right after a data gap) get NaN features.
Per block (summarising all its columns):
    _stale_max_log   log(1 + longest time since any column last changed)  -> "signal froze"
    _n_changes_win   how many value changes happened in the window         -> "signal went quiet"

The last two exist because in the BEAM results shown so far, the rows listed had their last
change more than an hour before the SBL, and the row-to-row differences were SMALLER than usual
in the 5 minutes before SBL - features that only look at value statistics would miss a signal
that simply stops updating. (Check on all 128 events whether this holds before relying on it.)
"""
import numpy as np
import pandas as pd


def window_seconds(cfg: dict, spacing_s: float) -> float:
    return max(pd.Timedelta(cfg["window"]).total_seconds(), cfg["min_rows_per_window"] * spacing_s)


def build_features(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """d: one preprocessed block (DatetimeIndex, numeric columns). Returns the feature table
    on the same index (rows without enough history contain NaN)."""
    spacing = d.index.to_series().diff().dt.total_seconds().median()
    win_s = window_seconds(cfg, spacing)
    win = pd.Timedelta(seconds=win_s)

    roll = d.rolling(win, min_periods=cfg["min_periods"])
    mx, mn = roll.max(), roll.min()

    # value one window ago (as-of), for the slope; rows with no history get slope 0
    t_ns = d.index.values.astype("datetime64[ns]").astype("int64")
    pos = np.searchsorted(t_ns, t_ns - int(win_s * 1e9), side="right") - 1
    vals = d.to_numpy(float)
    slope = np.where((pos >= 0)[:, None], (vals - vals[np.clip(pos, 0, None)]) / win_s, 0.0)

    feats = pd.concat({"val": d, "mean": roll.mean(), "std": roll.std(), "range": mx - mn,
                       "slope": pd.DataFrame(slope, index=d.index, columns=d.columns)}, axis=1)
    feats.columns = [f"{c}__{k}" for k, c in feats.columns]

    # staleness / activity of the whole block
    chg = d.diff().ne(0)
    chg.iloc[0] = False
    t = d.index.to_series()
    since_start = (t - t.iloc[0]).dt.total_seconds()
    stale = pd.DataFrame({c: (t - t.where(chg[c]).ffill()).dt.total_seconds() for c in d.columns})
    stale = stale.apply(lambda col: col.fillna(since_start))
    feats["_stale_max_log"] = np.log1p(stale.max(axis=1))
    feats["_n_changes_win"] = chg.sum(axis=1).rolling(win, min_periods=1).sum()

    # rows whose backward window is not fully covered by data (start of the series, or right after a
    # gap, e.g. after the non-BEAM_ON periods were removed) would carry truncated statistics that
    # look like anomalies -> mark them NaN so they are never scored
    gap_thr = max(pd.Timedelta(seconds=max(3 * spacing, 30)), pd.Timedelta(cfg["feature_gap"]))
    seg = (t.diff() > gap_thr).cumsum()
    age_s = (t - t.groupby(seg).transform("min")).dt.total_seconds()
    feats.loc[(age_s < win_s).to_numpy()] = np.nan

    feats.attrs["window_s"] = win_s
    feats.attrs["spacing_s"] = float(spacing)
    return feats.replace([np.inf, -np.inf], np.nan)
