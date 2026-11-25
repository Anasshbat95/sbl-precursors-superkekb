"""
sbl_labeling.py - Turn raw per-block rows into risk / normal labels, using the
four time windows around each SBL event:

    Risk       [SBL - risk_start,   SBL - buffer_start]  -> target = 1
    Buffer     [SBL - buffer_start, SBL]                 -> dropped
    Recovery   [SBL,                SBL + recovery_end]  -> dropped
    Normal     more than normal_margin away from EVERY SBL -> target = 0
    (unlabeled gray zone between recovery_end and normal_margin is dropped too)

Usage:
    from sbl_labeling import add_sbl_features_and_label

    beam_ds = add_sbl_features_and_label(beam, "timestamp", sbl_times["timestamp"])
"""
import numpy as np
import pandas as pd


def add_sbl_features_and_label(
    df: pd.DataFrame,
    ts_col: str,
    sbl_times,
    risk_start: pd.Timedelta = pd.Timedelta("5min"),
    buffer_start: pd.Timedelta = pd.Timedelta("30s"),
    recovery_end: pd.Timedelta = pd.Timedelta("10min"),
    normal_margin: pd.Timedelta = pd.Timedelta("15min"),
) -> pd.DataFrame:
    """Add 'sbl_diff_sec', 'min_abs_dist' and 'target' (1/0/NaN), then drop unlabeled rows.

    sbl_diff_sec: signed distance in seconds to the nearest SBL (negative = before, positive = after).
    min_abs_dist: unsigned distance in seconds to the nearest SBL.
    """
    df = df.copy()
    sbl_sorted = pd.to_datetime(pd.Series(sbl_times)).sort_values().values
    ts = df[ts_col].values

    idx = np.searchsorted(sbl_sorted, ts)
    idx_prev = np.clip(idx - 1, 0, len(sbl_sorted) - 1)
    idx_next = np.clip(idx, 0, len(sbl_sorted) - 1)

    delta_before = (sbl_sorted[idx_next] - ts) / np.timedelta64(1, "s")  # > 0 = before the next SBL
    delta_after = (ts - sbl_sorted[idx_prev]) / np.timedelta64(1, "s")   # > 0 = after the previous SBL

    df["sbl_diff_sec"] = np.minimum(delta_before, delta_after) * np.where(delta_before <= delta_after, 1, -1)
    df["min_abs_dist"] = np.minimum(np.abs(delta_before), np.abs(delta_after))

    label = np.full(len(df), np.nan)
    risk_mask = (delta_before >= buffer_start.total_seconds()) & (delta_before <= risk_start.total_seconds())
    buffer_mask = (delta_before < buffer_start.total_seconds()) & (delta_before >= 0)
    recovery_mask = (delta_after >= 0) & (delta_after <= recovery_end.total_seconds())
    normal_mask = df["min_abs_dist"] > normal_margin.total_seconds()

    label[risk_mask] = 1
    label[normal_mask] = 0
    df["target"] = label
    df.loc[buffer_mask | recovery_mask, "target"] = np.nan

    return df.dropna(subset=["target"]).reset_index(drop=True)


def label_all_blocks(blocks: dict[str, pd.DataFrame], sbl_times, ts_col: str = "timestamp", **kwargs) -> dict[str, pd.DataFrame]:
    """Apply add_sbl_features_and_label to every block in a {name: df} dict. Blocks stay separate."""
    return {name: add_sbl_features_and_label(df, ts_col, sbl_times, **kwargs) for name, df in blocks.items()}
