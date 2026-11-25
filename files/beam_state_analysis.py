"""
beam_state_analysis.py - SBLs and all data blocks split by beam state (beam ON / beam OFF).

You give the thresholds above which the HER and LER currents count as "beam present". From the beam
block every row gets a state per ring and a combined state:
    both_on   HER and LER above their thresholds
    her_only  only HER above its threshold
    ler_only  only LER above its threshold
    both_off  neither ring above its threshold
    unknown   (for SBLs / other blocks) no beam row close enough in time to decide
The binary "beam on" follows `on_rule`: "any" (either ring, default), "both", "her" or "ler".

What it answers:
    * how many SBLs happen while the beam is on, and how many while it is off (state taken from the last
      beam row BEFORE the SBL, so the loss itself is not part of it) - also 1 min and 5 min before
    * how much time (hours) the data spend in each state, and SBLs per beam-on hour
    * for every other block: how many rows lie in beam-on / beam-off / unknown, and how each feature's
      median, quartiles and share of exact zeros differ between on and off

An SBL cannot happen without beam. SBLs in "both_off" or "unknown" therefore usually point to a time
offset between the SBL list and the data (run check_time_offset in current_around_sbl.py) or to gaps in
the beam data - the summary tells you when that share is large.

Usage:
    from beam_state_analysis import (analyze_sbl_by_beam_state, plot_current_histogram,
                                     plot_sbl_beam_state, plot_block_state_shares)

    plot_current_histogram(ds_beam, "timestamp", "her_current", "ler_current", her_thr=50, ler_thr=50)
    res = analyze_sbl_by_beam_state(
        {"beam": ds_beam, "background": ds_background, "injection": ds_injection,
         "vacuum": ds_vacuum, "position": ds_position},
        sbl_times, beam_block="beam", her_thr=50, ler_thr=50, on_rule="any")
    res["summary"]; res["sbl_table"]; res["block_summary"]; res["feature_table"]
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATE_ORDER = ["both_on", "her_only", "ler_only", "both_off", "unknown"]
STATE_COLORS = {"both_on": "#2ca02c", "her_only": "#ff7f0e", "ler_only": "#1f77b4",
                "both_off": "#7f7f7f", "unknown": "#d9d9d9"}


def _sbl_series(sbl_times) -> pd.Series:
    if isinstance(sbl_times, pd.DataFrame):
        if sbl_times.shape[1] != 1:
            raise ValueError("sbl_times DataFrame must have exactly one column.")
        sbl_times = sbl_times.iloc[:, 0]
    return pd.to_datetime(pd.Series(sbl_times)).sort_values().reset_index(drop=True)


def _ns(times) -> np.ndarray:
    return pd.to_datetime(pd.Series(times)).to_numpy(dtype="datetime64[ns]").astype("int64")


def classify_beam_state(beam: pd.DataFrame, ts_col="timestamp", her_col="her_current", ler_col="ler_current",
                        her_thr=50.0, ler_thr=50.0, on_rule="any") -> pd.DataFrame:
    """State of every row of the beam block: her_on, ler_on, state (4 classes) and beam_on (binary)."""
    if on_rule not in ("any", "both", "her", "ler"):
        raise ValueError("on_rule must be 'any', 'both', 'her' or 'ler'.")
    d = beam[[ts_col, her_col, ler_col]].dropna().sort_values(ts_col).reset_index(drop=True)
    her_on, ler_on = d[her_col] > her_thr, d[ler_col] > ler_thr
    state = np.select([her_on & ler_on, her_on, ler_on], ["both_on", "her_only", "ler_only"], default="both_off")
    on = {"any": her_on | ler_on, "both": her_on & ler_on, "her": her_on, "ler": ler_on}[on_rule]
    return pd.DataFrame({ts_col: d[ts_col], "her": d[her_col], "ler": d[ler_col], "her_on": her_on,
                         "ler_on": ler_on, "state": state, "beam_on": on})


def _max_age_s(states: pd.DataFrame, ts_col: str, max_age) -> float:
    if max_age != "auto":
        return pd.Timedelta(max_age).total_seconds()
    spacing = states[ts_col].diff().dt.total_seconds().median()
    return max(60.0, 3 * float(spacing))


def beam_state_at(states: pd.DataFrame, times, ts_col="timestamp", max_age="auto", strict=False) -> pd.DataFrame:
    """Beam state at arbitrary times: the state of the last beam row at (strict=False) or strictly before
    (strict=True) each time. If that row is older than max_age (default "auto" = 3 median row spacings, at
    least 60 s) the state is "unknown". Returns state, beam ("on" / "off" / "unknown"), her, ler, age_s."""
    ts = _ns(states[ts_col])
    t = _ns(times)
    pos = np.searchsorted(ts, t, side="left" if strict else "right") - 1
    has = pos >= 0
    p = np.clip(pos, 0, None)
    age = np.where(has, (t - ts[p]) / 1e9, np.nan)
    ok = has & (age <= _max_age_s(states, ts_col, max_age))
    out = pd.DataFrame({"state": np.where(ok, states["state"].to_numpy()[p], "unknown"),
                        "beam": np.where(ok, np.where(states["beam_on"].to_numpy()[p], "on", "off"), "unknown"),
                        "her": np.where(ok, states["her"].to_numpy()[p], np.nan),
                        "ler": np.where(ok, states["ler"].to_numpy()[p], np.nan),
                        "age_s": age})
    return out


def state_hours(states: pd.DataFrame, ts_col="timestamp", max_gap="30min") -> pd.Series:
    """Hours the beam block spends in each of the 4 states: every row counts until the next row,
    capped at max_gap so that a data hole is not counted as a long period in one state."""
    dt = states[ts_col].shift(-1) - states[ts_col]
    dt = dt.dt.total_seconds().clip(upper=pd.Timedelta(max_gap).total_seconds()).fillna(0)
    return (dt.groupby(states["state"]).sum() / 3600).reindex(STATE_ORDER[:4]).fillna(0)


def sbl_beam_state_table(states: pd.DataFrame, sbl_times, ts_col="timestamp", max_age="auto",
                         lags=("1min", "5min")) -> pd.DataFrame:
    """One row per SBL: the beam state from the last beam row BEFORE the event (plus the on/off state
    `lags` earlier) and the age of that row."""
    sbl = _sbl_series(sbl_times)
    main = beam_state_at(states, sbl, ts_col, max_age, strict=True)
    tab = pd.DataFrame({"sbl_index": np.arange(len(sbl)), "sbl_time": sbl, "state": main["state"],
                        "beam": main["beam"], "her_before": main["her"], "ler_before": main["ler"],
                        "last_beam_row_age_s": main["age_s"]})
    for lag in lags:
        tab[f"beam_{lag}_before"] = beam_state_at(states, sbl - pd.Timedelta(lag), ts_col, max_age, strict=True)["beam"]
    return tab


def _block_analysis(name, df, states, ts_col, max_age, features=None):
    st = beam_state_at(states, df[ts_col], ts_col, max_age, strict=False)
    cols = features or [c for c in df.select_dtypes("number").columns if c != ts_col]
    n = len(df)
    counts = st["beam"].value_counts()
    summary = dict(block=name, rows=n, rows_on=int(counts.get("on", 0)), rows_off=int(counts.get("off", 0)),
                   rows_unknown=int(counts.get("unknown", 0)))
    summary["pct_on"] = round(100 * summary["rows_on"] / n, 1) if n else np.nan
    summary["pct_off"] = round(100 * summary["rows_off"] / n, 1) if n else np.nan
    summary["pct_unknown"] = round(100 * summary["rows_unknown"] / n, 1) if n else np.nan

    rows = []
    for c in cols:
        r = dict(block=name, feature=c)
        for s in ("on", "off"):
            v = df.loc[(st["beam"] == s).to_numpy(), c].dropna()
            r[f"n_{s}"] = len(v)
            r[f"median_{s}"] = float(v.median()) if len(v) else np.nan
            r[f"q25_{s}"] = float(v.quantile(0.25)) if len(v) else np.nan
            r[f"q75_{s}"] = float(v.quantile(0.75)) if len(v) else np.nan
            r[f"frac_zero_{s}"] = float((v == 0).mean()) if len(v) else np.nan
        rows.append(r)
    return summary, pd.DataFrame(rows)


def analyze_sbl_by_beam_state(blocks: dict, sbl_times, beam_block="beam", ts_col="timestamp",
                              her_col="her_current", ler_col="ler_current", her_thr=50.0, ler_thr=50.0,
                              on_rule="any", max_age="auto", max_gap="30min", features=None,
                              plot=True, dpi=300) -> dict:
    """Run the whole analysis (see the module docstring). `blocks` must contain the beam block under
    the name `beam_block`; every block (including the beam block itself) is analysed.
    `features` optionally limits the columns per block: {"background": ["her_tune_x", ...]}.
    Returns {"states", "sbl_table", "summary", "block_summary", "feature_table"}."""
    beam = blocks[beam_block]
    sbl = _sbl_series(sbl_times)
    tz_beam = pd.to_datetime(beam[ts_col]).dt.tz is not None
    if tz_beam != (sbl.dt.tz is not None):
        raise ValueError("Beam block and SBL times differ in time-zone awareness - convert both to the same "
                         "(e.g. both tz-naive UTC) first.")
    states = classify_beam_state(beam, ts_col, her_col, ler_col, her_thr, ler_thr, on_rule)
    tab = sbl_beam_state_table(states, sbl, ts_col, max_age)
    hours = state_hours(states, ts_col, max_gap)

    counts = tab["state"].value_counts().reindex(STATE_ORDER).fillna(0).astype(int)
    summary = pd.DataFrame({"n_sbl": counts, "pct_sbl": (100 * counts / max(len(tab), 1)).round(1)})
    summary["hours"] = hours.reindex(STATE_ORDER).fillna(0).round(1)
    summary["pct_time"] = (100 * summary["hours"] / max(summary["hours"].sum(), 1e-9)).round(1)
    summary["sbl_per_hour"] = (summary["n_sbl"] / summary["hours"].replace(0, np.nan)).round(4)

    bs, ft = [], []
    for name, df in blocks.items():
        s_, f_ = _block_analysis(name, df, states, ts_col, max_age, (features or {}).get(name))
        bs.append(s_)
        ft.append(f_)
    res = dict(states=states, sbl_table=tab, summary=summary, block_summary=pd.DataFrame(bs),
               feature_table=pd.concat(ft, ignore_index=True), her_thr=her_thr, ler_thr=ler_thr, on_rule=on_rule)

    # ---- printed report -----------------------------------------------------------------------------
    n = len(tab)
    on, off, unk = (int((tab["beam"] == k).sum()) for k in ("on", "off", "unknown"))
    print(f"Beam state before the SBL ({n} events; ON rule '{on_rule}', HER > {her_thr}, LER > {ler_thr}):")
    print(f"  beam ON : {on:4d}  ({100 * on / n:.0f}%)")
    print(f"  beam OFF: {off:4d}  ({100 * off / n:.0f}%)")
    print(f"  unknown : {unk:4d}  ({100 * unk / n:.0f}%)   (no beam row within {_max_age_s(states, ts_col, max_age):.0f} s before the SBL)")
    print("\nBy ring state, with the time the data spend in each state:")
    print(summary.to_string())
    print("\nRows of every block by beam state:")
    print(res["block_summary"].to_string(index=False))
    if (off + unk) > 0.1 * n:
        print(f"\nWARNING: {100 * (off + unk) / n:.0f}% of the SBLs have no beam before them (OFF or unknown). "
              "An SBL needs stored beam, so first check the clock between the SBL list and the data "
              "(current_around_sbl.check_time_offset), gaps in the beam block (last_beam_row_age_s in "
              "res['sbl_table']) and the thresholds (plot_current_histogram).")
    if plot:
        plot_sbl_beam_state(res, dpi=dpi)
        plot_block_state_shares(res, dpi=dpi)
    return res


def plot_current_histogram(beam: pd.DataFrame, ts_col="timestamp", her_col="her_current", ler_col="ler_current",
                           her_thr=50.0, ler_thr=50.0, bins=120, dpi=300):
    """Histogram of the HER and LER current values (log count axis) with the thresholds as dashed lines -
    use it to choose a threshold: above the noise at ~0, far below the operating level."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=dpi, layout="constrained")
    for ax, col, thr, name in ((axes[0], her_col, her_thr, "HER"), (axes[1], ler_col, ler_thr, "LER")):
        v = beam[col].dropna().to_numpy(float)
        ax.hist(v, bins=bins, color="#1f77b4" if name == "LER" else "#ff7f0e", alpha=0.8)
        ax.set_yscale("log")
        ax.axvline(thr, color="red", ls="--", lw=1.5, label=f"threshold = {thr:g}")
        ax.set_title(f"{name} current  ({col})", fontsize=10)
        ax.set_xlabel("current")
        ax.set_ylabel("number of rows (log)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.show()
    return fig


def plot_sbl_beam_state(res: dict, dpi=300):
    """Left: number of SBLs per beam state. Right: share of SBLs vs. share of the time per state - if
    the SBLs were independent of the state, both bars of a state would be equally high."""
    s = res["summary"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi, layout="constrained")
    colors = [STATE_COLORS[k] for k in s.index]
    bars = a1.bar(s.index, s["n_sbl"], color=colors, edgecolor="black")
    for b, v in zip(bars, s["n_sbl"]):
        a1.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(v)), ha="center", va="bottom", fontsize=9)
    a1.set_ylabel("number of SBLs")
    a1.set_title(f"SBLs by beam state before the event (n = {int(s['n_sbl'].sum())})", fontsize=10)
    x = np.arange(len(s))
    a2.bar(x - 0.2, s["pct_sbl"], width=0.4, color="firebrick", label="% of the SBLs")
    a2.bar(x + 0.2, s["pct_time"], width=0.4, color="steelblue", label="% of the time")
    a2.set_xticks(x)
    a2.set_xticklabels(s.index)
    a2.set_ylabel("percent")
    a2.set_title("share of SBLs vs. share of time in each state", fontsize=10)
    a2.legend(fontsize=8)
    for ax in (a1, a2):
        ax.grid(axis="y", alpha=0.3)
    plt.show()
    return fig


def plot_block_state_shares(res: dict, dpi=300):
    """Stacked bars: share of the rows of every block that lie in beam ON / OFF / unknown."""
    b = res["block_summary"].set_index("block")
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(b) + 1.5), dpi=dpi, layout="constrained")
    left = np.zeros(len(b))
    for col, label, color in (("pct_on", "beam on", "#2ca02c"), ("pct_off", "beam off", "#7f7f7f"),
                              ("pct_unknown", "unknown", "#d9d9d9")):
        ax.barh(b.index, b[col], left=left, color=color, edgecolor="black", label=label)
        left += b[col].to_numpy()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the rows of the block")
    ax.set_title("Rows per block by beam state", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    ax.invert_yaxis()
    plt.show()
    return fig
