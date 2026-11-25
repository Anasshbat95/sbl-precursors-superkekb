import numpy as np
import pandas as pd


def changes_per_sbl(df, sbl_times, features, window_seconds=300, buffer_seconds=30, ts_col="timestamp"):
    """One row per SBL: number of value changes of each feature in [SBL - window, SBL - buffer).
    Shows whether a high enrichment comes from many SBLs or from a few."""
    d = df.sort_values(ts_col)
    t = pd.to_datetime(d[ts_col]).to_numpy(dtype="datetime64[ns]").astype("int64")
    s = np.sort(pd.to_datetime(pd.Series(sbl_times)).to_numpy(dtype="datetime64[ns]").astype("int64"))
    lo, hi = s - int(window_seconds * 1e9), s - int(buffer_seconds * 1e9)
    out = pd.DataFrame({"sbl_time": pd.to_datetime(s)})
    for f in features:
        v = d[f]
        ch = (v.ne(v.shift()) & v.notna()).to_numpy().copy()
        ch[0] = False
        tc = t[ch]
        out[f] = np.searchsorted(tc, hi, "left") - np.searchsorted(tc, lo, "left")
    summary = pd.DataFrame({f: {"sbl_with_change": int((out[f] > 0).sum()), "total_changes": int(out[f].sum()),
                                "max_in_one_sbl": int(out[f].max())} for f in features}).T
    return out, summary
