"""
anomaly_labeling.py - Prepare a block for ANOMALY detection (not classification).

Unlike sbl_labeling.add_sbl_features_and_label, this function:
  - keeps every row (no dropna) so the time series stays continuous for
    window-based models (see sbl_anomaly.make_windows, which relies on
    consecutive rows and only splits on real time gaps),
  - does NOT create a 0/1 'target' column. Anomaly models train only on
    "normal" rows; the risk window is not a training label, it is the
    thing you evaluate detection against afterwards.

It adds:
    sbl_diff_sec   signed distance in seconds to the nearest SBL (negative = before, positive = after)
    min_abs_dist   unsigned distance in seconds to the nearest SBL
    is_normal      True if farther than normal_margin from EVERY SBL (safe to use for training/calibration)
    phase          human-readable label for inspection/plots: "risk", "buffer", "recovery", "normal", "gray_zone"
                   (informational only - the anomaly pipeline uses is_normal, not phase)

Usage:
    from anomaly_labeling import add_sbl_distance_features

    beam_ds = add_sbl_distance_features(beam, "timestamp", sbl_times["timestamp"])
    train_rows = beam_ds[beam_ds["is_normal"]]      # use only for fitting mean/std/model
    # beam_ds itself (all rows, in order) still goes into sbl_anomaly.run(...)
"""
import numpy as np
import pandas as pd


def add_sbl_distance_features(
    df: pd.DataFrame,
    ts_col: str,
    sbl_times,
    risk_start: pd.Timedelta = pd.Timedelta("5min"),
    buffer_start: pd.Timedelta = pd.Timedelta("30s"),
    recovery_end: pd.Timedelta = pd.Timedelta("10min"),
    normal_margin: pd.Timedelta = pd.Timedelta("15min"),
) -> pd.DataFrame:
    """Add distance-to-SBL columns and an is_normal flag, without dropping or re-labeling any row."""
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
    df["is_normal"] = df["min_abs_dist"] > normal_margin.total_seconds()

    phase = np.full(len(df), "gray_zone", dtype=object)
    risk_mask = (delta_before >= buffer_start.total_seconds()) & (delta_before <= risk_start.total_seconds())
    buffer_mask = (delta_before < buffer_start.total_seconds()) & (delta_before >= 0)
    recovery_mask = (delta_after >= 0) & (delta_after <= recovery_end.total_seconds())
    phase[risk_mask] = "risk"
    phase[buffer_mask] = "buffer"
    phase[recovery_mask] = "recovery"
    phase[df["is_normal"].to_numpy()] = "normal"
    df["phase"] = phase

    return df


def prepare_all_blocks(blocks: dict[str, pd.DataFrame], sbl_times, ts_col: str = "timestamp",
                       **kwargs) -> dict[str, pd.DataFrame]:
    """Apply add_sbl_distance_features to every block in a {name: df} dict. Blocks stay separate, no rows dropped."""
    return {name: add_sbl_distance_features(df, ts_col, sbl_times, **kwargs) for name, df in blocks.items()}
