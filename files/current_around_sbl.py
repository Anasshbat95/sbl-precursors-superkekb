"""
current_around_sbl.py - How do the HER and LER beam currents behave before and after each SBL?

Per event (one row per SBL) the table answers:
    * was the beam ON right before the SBL, in which ring, and for how long had it been on?
    * which ring lost its beam (HER, LER, both, none) - from the drop in current right after the event
    * how long after the SBL was the current below the threshold (beam OFF)?
    * when did the beam come back (current above the threshold again for min_on)?
    * how good is the data around the event (age of the last row before, delay of the first row after)?

Also: check_time_offset() tests whether the SBL timestamps are on the same clock as the beam data
(e.g. German local time vs UTC): with the right clock, the current drops right AFTER the SBL.

Usage:
    from current_around_sbl import (current_around_sbl, summarize_current_events,
                                    check_time_offset, plot_current_around_sbl)

    tab = current_around_sbl(ds_beam, "timestamp", sbl_times,
                             her_col="her_current", ler_col="ler_current", thr=50)
    summarize_current_events(tab)
    check_time_offset(ds_beam, "timestamp", sbl_times)          # run this first if the summary looks odd
    plot_current_around_sbl(ds_beam, "timestamp", sbl_times, before="30min", after="60min")

`thr` is the current (same unit as your data) above which a ring counts as "beam present". Choose it
from the histogram of the current values (see the note in the chat): above the noise floor near 0 and
far below the operating level. It can be a single number or {"HER": 50, "LER": 80}.
"""
import numpy as np
import pandas as pd


def _sbl_series(sbl_times) -> pd.Series:
    if isinstance(sbl_times, pd.DataFrame):
        if sbl_times.shape[1] != 1:
            raise ValueError("sbl_times DataFrame must have exactly one column.")
        sbl_times = sbl_times.iloc[:, 0]
    return pd.to_datetime(pd.Series(sbl_times)).sort_values().reset_index(drop=True).astype("datetime64[ns]")


def _prepare(beam, ts_col, her_col, ler_col):
    d = beam[[ts_col, her_col, ler_col]].dropna().sort_values(ts_col)
    ts = pd.to_datetime(d[ts_col]).to_numpy(dtype="datetime64[ns]")
    return ts, {"HER": d[her_col].to_numpy(float), "LER": d[ler_col].to_numpy(float)}


def _asof(ts, vals, t):
    """Last known value at or before t (NaN if there is none)."""
    i = np.searchsorted(ts, t, side="right") - 1
    return vals[i] if i >= 0 else np.nan


def _first_sustained_on(ts, above, start_idx, end_idx, min_on_ns):
    """Index of the first row in [start_idx, end_idx) that begins a run of 'above' lasting >= min_on."""
    i = start_idx
    while i < end_idx:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < end_idx and above[j]:
            j += 1
        run_end = ts[j] if j < end_idx else ts[end_idx - 1]
        if (run_end - ts[i]).astype("int64") >= min_on_ns or j >= end_idx:
            return i
        i = j
    return None


def current_around_sbl(beam, ts_col, sbl_times, her_col="her_current", ler_col="ler_current",
                       thr=50.0, post="60min", min_on="2min", drop_frac=0.5, check_after="5min",
                       gap="30min") -> pd.DataFrame:
    """One row per SBL event. Times in seconds unless the column name says otherwise.

    post         how far after the SBL to look for "beam off" and "beam on again"
    min_on       the current must stay above thr this long to count as "beam on again"
    drop_frac    a ring counts as "lost" if its current within `check_after` after the SBL falls below
                 drop_frac x the value just before the SBL
    gap          a hole in the data longer than this ends a continuous "on" period
    """
    thr = {"HER": thr, "LER": thr} if np.isscalar(thr) else dict(thr)
    ts, cur = _prepare(beam, ts_col, her_col, ler_col)
    sbl = _sbl_series(sbl_times)
    n = len(ts)
    if n == 0:
        raise ValueError("No rows with both current columns present.")
    post_ns = pd.Timedelta(post).value
    min_on_ns = pd.Timedelta(min_on).value
    chk_ns = pd.Timedelta(check_after).value
    seg = np.r_[0, np.cumsum(np.diff(ts).astype("int64") > pd.Timedelta(gap).value)]
    seg_start = np.zeros(n, int)
    seg_start[np.r_[True, seg[1:] != seg[:-1]]] = np.arange(n)[np.r_[True, seg[1:] != seg[:-1]]]
    seg_start = np.maximum.accumulate(seg_start)

    rows = []
    for k, t in enumerate(sbl):
        tn = np.datetime64(t, "ns")
        ib = np.searchsorted(ts, tn, side="left") - 1          # last row before the SBL
        ia = ib + 1                                              # first row at/after the SBL
        end = np.searchsorted(ts, tn + np.timedelta64(post_ns, "ns"), side="right")
        r = dict(sbl_index=k, sbl_time=t,
                 last_row_before_age_s=(tn - ts[ib]).astype("int64") / 1e9 if ib >= 0 else np.nan,
                 first_row_after_delay_s=(ts[ia] - tn).astype("int64") / 1e9 if ia < n else np.nan)
        lost = []
        for ring in ("HER", "LER"):
            v, th = cur[ring], thr[ring]
            before = v[ib] if ib >= 0 else np.nan
            r[f"{ring}_before"] = before
            r[f"{ring}_m10min"] = _asof(ts, v, tn - np.timedelta64(pd.Timedelta("10min").value, "ns"))
            r[f"{ring}_p1min"] = _asof(ts, v, tn + np.timedelta64(pd.Timedelta("1min").value, "ns"))
            r[f"{ring}_p10min"] = _asof(ts, v, tn + np.timedelta64(pd.Timedelta("10min").value, "ns"))
            on_before = bool(before > th) if not np.isnan(before) else False
            r[f"{ring}_on_before"] = on_before

            # how long has this ring been continuously on before the SBL?
            on_min = np.nan
            if on_before:
                below_idx = np.where(v[:ib + 1] <= th)[0]
                start = max(below_idx[-1] + 1 if len(below_idx) else 0, seg_start[ib])
                on_min = (tn - ts[start]).astype("int64") / 6e10
                r[f"{ring}_on_since_data_start"] = (len(below_idx) == 0 and seg_start[ib] == 0)
            r[f"{ring}_on_min_before"] = on_min

            # what happens after the SBL?
            off_delay = restart_min = np.nan
            dropped = np.nan
            if on_before and ia < end:
                sub = np.arange(ia, end)
                chk = sub[(ts[sub] - tn).astype("int64") <= chk_ns]
                if len(chk):
                    dropped = bool(v[chk].min() < drop_frac * before)
                below = sub[v[sub] <= th]
                if len(below):
                    off_i = below[0]
                    off_delay = (ts[off_i] - tn).astype("int64") / 1e9
                    j = _first_sustained_on(ts, v > th, off_i, end, min_on_ns)
                    if j is not None:
                        restart_min = (ts[j] - tn).astype("int64") / 6e10
            r[f"{ring}_dropped"] = dropped
            r[f"{ring}_off_delay_s"] = off_delay
            r[f"{ring}_restart_min"] = restart_min
            if dropped is True:
                lost.append(ring)
        both_known = not (np.isnan(r["HER_dropped"]) and np.isnan(r["LER_dropped"]))
        r["ring_lost"] = ("both" if len(lost) == 2 else lost[0] if lost else
                          ("none" if both_known else "unknown"))
        r["any_ring_on_before"] = r["HER_on_before"] or r["LER_on_before"]
        rows.append(r)
    return pd.DataFrame(rows)


def summarize_current_events(tab: pd.DataFrame) -> None:
    n = len(tab)
    print(f"{n} SBL events")
    print("\nBeam present right before the SBL (last row before the event):")
    for ring in ("HER", "LER"):
        print(f"  {ring}: on in {int(tab[f'{ring}_on_before'].sum())}/{n}")
    n_none = int((~tab["any_ring_on_before"]).sum())
    print(f"  no ring on: {n_none}/{n}" + ("   <-- an SBL cannot happen without beam: check the clock "
          "(check_time_offset) and the data gaps below" if n_none > 0.1 * n else ""))
    print("\nWhich ring lost its beam right after the SBL:")
    print(tab["ring_lost"].value_counts().to_string())
    print("\nData around the events (seconds):")
    print(tab[["last_row_before_age_s", "first_row_after_delay_s"]].describe().loc[["count", "50%", "max"]].round(1).to_string())
    for ring in ("HER", "LER"):
        sub = tab[tab[f"{ring}_dropped"] == True]  # noqa: E712
        if len(sub):
            print(f"\n{ring} events where {ring} dropped ({len(sub)}):")
            print(sub[[f"{ring}_on_min_before", f"{ring}_off_delay_s", f"{ring}_restart_min"]]
                  .describe().loc[["count", "25%", "50%", "75%"]].round(1).to_string())
    print("\n(on_min_before = minutes the ring had been continuously on before the SBL; off_delay_s = seconds "
          "until the current was below thr; restart_min = minutes after the SBL until the beam was on again.)")


def check_time_offset(beam, ts_col, sbl_times, her_col="her_current", ler_col="ler_current", thr=50.0,
                      shifts=None, drop_frac=0.5) -> pd.DataFrame:
    """Shift all SBL times by each candidate offset and count how often a ring that had beam in the
    5 min before (until 30 s before) the shifted time has less than drop_frac of it in the 5 min after
    (from 30 s after). With the correct clock this count is clearly the highest. `shift` is what you
    would ADD to your SBL times: a best shift of -1h means your SBL list is 1 hour ahead of the data."""
    thr = {"HER": thr, "LER": thr} if np.isscalar(thr) else dict(thr)
    ts, cur = _prepare(beam, ts_col, her_col, ler_col)
    sbl = _sbl_series(sbl_times)
    shifts = shifts if shifts is not None else [pd.Timedelta(minutes=m) for m in range(-180, 181, 30)]
    s30, s5 = pd.Timedelta("30s").value, pd.Timedelta("5min").value
    rows = []
    for sh in shifts:
        sh = pd.Timedelta(sh)
        with_data = drops = 0
        for t in sbl + sh:
            tn = np.datetime64(t, "ns")
            lo_b, hi_b = np.searchsorted(ts, [tn - np.timedelta64(s5, "ns"), tn - np.timedelta64(s30, "ns")])
            lo_a, hi_a = np.searchsorted(ts, [tn + np.timedelta64(s30, "ns"), tn + np.timedelta64(s5, "ns")])
            if hi_b - lo_b < 1 or hi_a - lo_a < 1:
                continue
            with_data += 1
            for ring in ("HER", "LER"):
                b = np.median(cur[ring][lo_b:hi_b])
                a = np.median(cur[ring][lo_a:hi_a])
                if b > thr[ring] and a < drop_frac * b:
                    drops += 1
                    break
        rows.append(dict(shift=f"{sh.total_seconds() / 3600:+.1f}h", events_with_data=with_data, events_with_drop=drops,
                         drop_fraction=round(drops / with_data, 3) if with_data else np.nan))
    out = pd.DataFrame(rows)
    best = out.loc[out["drop_fraction"].idxmax()] if out["drop_fraction"].notna().any() else None
    print(out.to_string(index=False))
    if best is not None:
        zero = out.loc[out["shift"] == "+0.0h", "drop_fraction"]
        print(f"\nHighest drop fraction at shift {best['shift']} ({best['drop_fraction']:.0%}); "
              f"at shift +0.0h it is {float(zero.iloc[0]):.0%}." if len(zero) else "")
        print("If the best shift is not +0.0h and clearly higher than the +0.0h value, the SBL list and the "
              "beam data are on different clocks (or the list has a different meaning).")
    return out


def plot_current_around_sbl(beam, ts_col, sbl_times, her_col="her_current", ler_col="ler_current",
                            thr=50.0, before="30min", after="60min", max_events=None):
    """Event-locked HER / LER current: x = minutes relative to the SBL (0 = SBL). Top row: raw current;
    bottom row: current divided by its value just before the SBL (shows the drop independent of the
    absolute level). Thin gray = one event, red = median over events, dashed = beam-on threshold."""
    import matplotlib.pyplot as plt

    thr = {"HER": thr, "LER": thr} if np.isscalar(thr) else dict(thr)
    ts, cur = _prepare(beam, ts_col, her_col, ler_col)
    sbl = _sbl_series(sbl_times)
    if max_events:
        sbl = sbl.iloc[:max_events]
    b_ns, a_ns = pd.Timedelta(before).value, pd.Timedelta(after).value

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for col, ring in enumerate(("HER", "LER")):
        pooled_raw, pooled_rel = [], []
        for t in sbl:
            tn = np.datetime64(t, "ns")
            lo = np.searchsorted(ts, tn - np.timedelta64(b_ns, "ns"))
            hi = np.searchsorted(ts, tn + np.timedelta64(a_ns, "ns"))
            if hi - lo < 2:
                continue
            x = (ts[lo:hi] - tn).astype("int64") / 6e10
            y = cur[ring][lo:hi]
            ref = _asof(ts, cur[ring], tn - np.timedelta64(1, "ns"))
            axes[0, col].plot(x, y, color="gray", alpha=0.25, lw=0.7)
            pooled_raw.append(pd.Series(y, index=np.round(x)))
            if ref and ref > thr[ring]:
                axes[1, col].plot(x, y / ref, color="gray", alpha=0.25, lw=0.7)
                pooled_rel.append(pd.Series(y / ref, index=np.round(x)))
        for row, pooled in ((0, pooled_raw), (1, pooled_rel)):
            if pooled:
                med = pd.concat(pooled).groupby(level=0).median()
                axes[row, col].plot(med.index, med.values, color="firebrick", lw=2, label="median")
            axes[row, col].axvline(0, color="red")
        axes[0, col].axhline(thr[ring], color="black", linestyle="--", lw=1)
        axes[0, col].set_title(f"{ring} current")
        axes[1, col].set_title(f"{ring} current / value just before SBL (only events with {ring} on)")
        axes[1, col].set_xlabel("minutes relative to SBL (0 = SBL)")
    axes[0, 0].set_ylabel("current")
    axes[1, 0].set_ylabel("relative current")
    plt.tight_layout()
    plt.show()
