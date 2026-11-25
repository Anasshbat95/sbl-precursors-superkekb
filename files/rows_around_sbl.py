import numpy as np
import pandas as pd


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _ns(x):
    return np.sort(pd.to_datetime(pd.Series(x)).to_numpy(dtype="datetime64[ns]").astype("int64"))


def rows_around_sbl(blocks, sbl_times, window="60s", shifts=("1h", "2h", "3h", "4h", "5h", "6h"), ts_col="timestamp"):
    """Per block: rows in [t-window, t) before each SBL, rows in the SBL second itself, rows in (t, t+window],
    compared with control times t-1h ... t-6h (same operating conditions, no SBL).
    Row level:   rows_before_per_sbl / rows_control_per_time = ratio_before_vs_control
    Event level: share of SBLs with at least one row before (Wilson 95% interval) vs. the same share at the
                 control times. Event level is the fairer comparison: a few SBLs with many rows cannot dominate it."""
    s = _ns(sbl_times)
    w, sec = pd.Timedelta(window).value, pd.Timedelta("1s").value
    rows = []
    for name, df in blocks.items():
        t = _ns(df[ts_col])
        cnt = lambda a, b: np.searchsorted(t, b, "left") - np.searchsorted(t, a, "left")   # rows in [a, b)
        before, at, after = cnt(s - w, s), cnt(s, s + sec), cnt(s + sec, s + w + sec)
        ctrl = np.concatenate([cnt(s - pd.Timedelta(x).value - w, s - pd.Timedelta(x).value) for x in shifts])
        k = int((before > 0).sum())
        lo, hi = _wilson(k, len(s))
        rows.append(dict(block=name,
                         rows_before_per_sbl=round(before.mean(), 3),
                         rows_control_per_time=round(ctrl.mean(), 3),
                         ratio_before_vs_control=round(before.mean() / ctrl.mean(), 2) if ctrl.mean() > 0 else np.nan,
                         sbl_with_rows_before=f"{k}/{len(s)}",
                         frac_sbl_with_rows=round(k / len(s), 3),
                         frac_sbl_ci95=f"{lo:.2f}-{hi:.2f}",
                         frac_control_with_rows=round(float((ctrl > 0).mean()), 3),
                         max_rows_one_sbl=int(before.max()),
                         rows_at_sbl_second=int(at.sum()),
                         rows_after_per_sbl=round(after.mean(), 3)))
    return pd.DataFrame(rows)


def rows_per_event(blocks, sbl_times, window="60s", ts_col="timestamp"):
    """One row per SBL: number of rows of every block in [t-window, t). Shows whether the extra rows come from
    many SBLs or from a few, and whether the same SBLs are busy in several blocks."""
    s = _ns(sbl_times)
    w = pd.Timedelta(window).value
    out = pd.DataFrame({"sbl_time": pd.to_datetime(s)})
    for name, df in blocks.items():
        t = _ns(df[ts_col])
        out[name] = np.searchsorted(t, s, "left") - np.searchsorted(t, s - w, "left")
    out["total"] = out[list(blocks)].sum(axis=1)
    return out.sort_values("total", ascending=False).reset_index(drop=True)
