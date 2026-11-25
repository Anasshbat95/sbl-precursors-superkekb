"""
operation_state.py - Determine the machine operating state from the beam current
and apply it to all 5 blocks (no merging of raw data).

States:
  BEAM_ON   beam present (max(HER, LER) > on_thr)
  RAMP_UP   first minutes after beam-on (only if rampup is set; default: off)
  BEAM_OFF  short pause (< shutdown_min), e.g. between fills / after an SBL
  SHUTDOWN  long pause (>= shutdown_min), e.g. winter shutdown
  NO_DATA   timestamp outside the known intervals

Usage:
    from operation_state import *
    iv = operation_intervals(BEAM, cols=("her_current", "ler_current"),
                             on_thr=400, shutdown_min="3D", rampup=None)
    print(summary(iv, sbl_times["timestamp"]))
    blocks_on = {n: label_states(df, iv).query("state == 'BEAM_ON'").drop(columns="state")
                 for n, df in blocks.items()}
    fig, ax = plot_states(BEAM, iv, title="...", save_as="figures/ch3_state_timeline.png")
    off_intervals = get_off_intervals(iv)
    fig, ax = plot_shutdown_timeline(off_intervals, save_as="figures/ch3_shutdown_intervals.png")
    fig, ax = plot_all_states_timeline(iv, save_as="figures/ch3_all_states.png")
"""
import numpy as np
import pandas as pd


def _merge(iv):
    iv = iv.sort_values("start").reset_index(drop=True)
    grp = (iv["on"] != iv["on"].shift()).cumsum()
    return iv.groupby(grp).agg(start=("start", "min"), end=("end", "max"), on=("on", "first")).reset_index(drop=True)


def operation_intervals(beam, ts_col="timestamp", cols=("her_current", "ler_current"),
                        on_thr=400.0, grid="1min", max_hold="2h", min_on="5min",
                        shutdown_min="3D", rampup=None, verbose=True):
    """Interval table (start, end, state): gapless and non-overlapping.

    on_thr       : beam counts as on if max(HER, LER) > on_thr (mA).
    shutdown_min : an off-phase at least this long is SHUTDOWN, shorter ones are BEAM_OFF.
    rampup       : None (default) -> no RAMP_UP state, every on-phase is BEAM_ON,
                   so no data directly before an SBL is cut away.
                   e.g. "15min" -> the first 15 min of each on-phase become RAMP_UP.
    """
    d = beam[[ts_col, *cols]].dropna()
    cur = pd.Series(d[list(cols)].max(axis=1).to_numpy(), index=pd.DatetimeIndex(d[ts_col])).sort_index()
    cur = cur.groupby(level=0).max()                                   # remove duplicate timestamps
    step = cur.index.to_series().diff().median()

    g = cur.resample(grid).max()                                       # NaN where no row falls in a bin
    g = g.ffill(limit=max(1, int(pd.Timedelta(max_hold) / pd.Timedelta(grid))))
    on = (g > on_thr).to_numpy()                                       # NaN (no data) -> False

    change = np.r_[True, on[1:] != on[:-1]]
    iv = (pd.DataFrame({"start": g.index, "on": on, "run": np.cumsum(change)})
            .groupby("run").agg(start=("start", "min"), on=("on", "first")))
    iv["end"] = iv["start"].shift(-1)
    iv.loc[iv.index[-1], "end"] = g.index[-1] + pd.Timedelta(grid)
    iv = _merge(iv[["start", "end", "on"]].reset_index(drop=True))

    # very short "on" blips are not an operating phase
    short_on = iv["on"] & ((iv["end"] - iv["start"]) < pd.Timedelta(min_on))
    iv.loc[short_on, "on"] = False
    iv = _merge(iv)

    dur = iv["end"] - iv["start"]
    iv["state"] = np.where(iv["on"], "BEAM_ON",
                           np.where(dur >= pd.Timedelta(shutdown_min), "SHUTDOWN", "BEAM_OFF"))

    if rampup is None or pd.Timedelta(rampup) <= pd.Timedelta(0):
        res = iv[["start", "end", "state"]].reset_index(drop=True)      # no RAMP_UP state
    else:
        # split off the ramp-up phase after each beam-on
        out = []
        for i, r in iv.iterrows():
            after_off = i > 0 and not iv["on"].iloc[i - 1]
            if r["state"] == "BEAM_ON" and after_off and (r["end"] - r["start"]) > pd.Timedelta(rampup):
                mid = r["start"] + pd.Timedelta(rampup)
                out.append((r["start"], mid, "RAMP_UP"))
                out.append((mid, r["end"], "BEAM_ON"))
            elif r["state"] == "BEAM_ON" and after_off:
                out.append((r["start"], r["end"], "RAMP_UP"))
            else:
                out.append((r["start"], r["end"], r["state"]))
        res = pd.DataFrame(out, columns=["start", "end", "state"])

    if verbose:
        hrs = ((res["end"] - res["start"]).dt.total_seconds() / 3600).groupby(res["state"]).sum().round(1)
        print(f"threshold: {on_thr} | shutdown >= {shutdown_min} | ramp-up: {rampup}")
        print(f"rows: {len(cur)} | median row spacing: {step} | rows above threshold: {(cur > on_thr).mean():.1%}")
        print("hours per state:", hrs.to_dict())
        if step > pd.Timedelta("1min"):
            print("WARNING: rows are more than 1 min apart. State boundaries are only as precise as the "
                  "sampling, and windows like 5 min / 30 s cannot be resolved with such data.")
    return res


def label_states(df, iv, ts_col="timestamp"):
    """Adds the column 'state' to a block (use the same iv for every block)."""
    idx = pd.IntervalIndex.from_arrays(iv["start"], iv["end"], closed="left")
    pos = idx.get_indexer(pd.DatetimeIndex(df[ts_col]))
    out = df.copy()
    out["state"] = np.where(pos >= 0, iv["state"].to_numpy()[np.clip(pos, 0, len(iv) - 1)], "NO_DATA")
    return out


def summary(iv, sbl_times):
    """Hours per state + the state each SBL was in (1 min before the event)."""
    hours = ((iv["end"] - iv["start"]).dt.total_seconds() / 3600).groupby(iv["state"]).agg(["sum", "count"])
    hours.columns = ["hours", "intervals"]
    idx = pd.IntervalIndex.from_arrays(iv["start"], iv["end"], closed="left")
    sbl = pd.DatetimeIndex(pd.to_datetime(pd.Series(sbl_times)))
    pos = idx.get_indexer(sbl - pd.Timedelta("1min"))
    states = pd.Series(np.where(pos >= 0, iv["state"].to_numpy()[np.clip(pos, 0, len(iv) - 1)], "NO_DATA"))
    hours["sbl_count"] = states.value_counts().reindex(hours.index).fillna(0).astype(int)
    return hours.round(1)


def plot_states(beam, iv, cols=("her_current", "ler_current"), ts_col="timestamp",
                title=None, save_as=None):
    """Beam current with shaded states - to check the threshold visually.

    title   : text above the plot (default: colour legend of the states)
    save_as : file name, e.g. "figures/ch3_state_timeline.png" (None = do not save)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = {"SHUTDOWN": "#b0b0b0", "BEAM_OFF": "#f4c542", "RAMP_UP": "#8ecae6", "BEAM_ON": "#90be6d"}
    fig, ax = plt.subplots(figsize=(14, 4))
    for c in cols:
        ax.plot(beam[ts_col], beam[c], lw=0.6, label=c)
    for _, r in iv.iterrows():
        if colors[r["state"]]:
            ax.axvspan(r["start"], r["end"], color=colors[r["state"]], alpha=0.35, lw=0)
    lines, _ = ax.get_legend_handles_labels()
    patches = [Patch(color=colors[s], alpha=0.35, label=s) for s in colors if s in set(iv["state"])]
    ax.set_ylabel("Beam current"); ax.legend(handles=lines + patches, loc="upper left")
    ax.set_title(title or "Operating states from the beam current")
    plt.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


def remove_shutdown(df, iv, ts_col="timestamp", states_to_remove=("SHUTDOWN",)):
    """Remove rows of `df` that fall into any interval whose state is in `states_to_remove`."""
    idx = pd.IntervalIndex.from_arrays(iv["start"], iv["end"], closed="left")
    pos = idx.get_indexer(pd.DatetimeIndex(df[ts_col]))
    state = np.where(pos >= 0, iv["state"].to_numpy()[np.clip(pos, 0, len(iv) - 1)], "NO_DATA")
    keep = ~np.isin(state, states_to_remove)
    return df[keep].reset_index(drop=True)


def get_all_intervals(iv: pd.DataFrame) -> pd.DataFrame:
    """Return every interval (all states) with an added 'duration' and 'duration_hours' column."""
    out = iv.copy()
    out["duration"] = out["end"] - out["start"]
    out["duration_hours"] = out["duration"].dt.total_seconds() / 3600
    return out.reset_index(drop=True)


def get_off_intervals(iv: pd.DataFrame) -> pd.DataFrame:
    """Return only the non-BEAM_ON intervals (BEAM_OFF, RAMP_UP, SHUTDOWN) with their duration."""
    off = get_all_intervals(iv)
    return off[off["state"] != "BEAM_ON"].reset_index(drop=True)


STATE_COLORS = {"SHUTDOWN": "#b0b0b0", "BEAM_OFF": "#f4c542", "RAMP_UP": "#8ecae6", "BEAM_ON": "#90be6d"}


def plot_shutdown_timeline(off_intervals, title=None, save_as=None):
    """Bar for every non-BEAM_ON interval (from get_off_intervals), length = duration.

    Each interval is one row; the label shows its duration, so it is easy to see
    which pauses count as SHUTDOWN (>= shutdown_min) and which as BEAM_OFF.
    By default only SHUTDOWN intervals are shown, because BEAM_OFF can be hundreds of rows.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    d = off_intervals[off_intervals["state"] == "SHUTDOWN"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.5 * max(len(d), 1)))
    for i, r in d.iterrows():
        ax.barh(i, r["end"] - r["start"], left=r["start"], color=STATE_COLORS[r["state"]], height=0.6)
        days = (r["end"] - r["start"]).total_seconds() / 86400
        ax.text(r["start"] + (r["end"] - r["start"]) / 2, i,
                f"{r['start']:%Y-%m-%d %H:%M} to {r['end']:%Y-%m-%d %H:%M} ({days:.1f} days)",
                ha="center", va="center", fontsize=9)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"SHUTDOWN {i + 1}" for i in range(len(d))])
    ax.invert_yaxis()
    ax.set_title(title or "SHUTDOWN intervals found from the beam current")
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_all_states_timeline(iv, title=None, save_as=None):
    """One horizontal band per state over time (no current), to see when each state occurs."""
    import matplotlib.pyplot as plt
    states = [s for s in ["BEAM_ON", "RAMP_UP", "BEAM_OFF", "SHUTDOWN"] if s in set(iv["state"])]
    fig, ax = plt.subplots(figsize=(14, 0.6 + 0.6 * len(states)))
    for k, s in enumerate(states):
        d = iv[iv["state"] == s]
        ax.broken_barh(list(zip(d["start"], d["end"] - d["start"])), (k - 0.35, 0.7),
                       facecolors=STATE_COLORS[s])
    hrs = ((iv["end"] - iv["start"]).dt.total_seconds() / 3600).groupby(iv["state"]).sum()
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([f"{s} ({hrs[s]:.0f} h)" for s in states])
    ax.invert_yaxis()
    ax.set_title(title or "Operating states over time")
    plt.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax