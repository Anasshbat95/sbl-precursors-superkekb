"""
operation_state.py - Determine the machine operating state from the beam current
and apply it to all 5 blocks (no merging of raw data).

States:
  BEAM_ON   beam present            -> train / label "normal" only here
  RAMP_UP   first minutes after beam-on (injection, oscillations) -> exclude
  BEAM_OFF  short pause (< shutdown_min), e.g. between fills / after an SBL
  SHUTDOWN  long pause (>= shutdown_min), e.g. winter shutdown
  NO_DATA   timestamp outside the known intervals
"""

import numpy as np
import pandas as pd


def _merge(iv):
  iv = iv.sort_values("start").reset_index(drop=True)
  grp = (iv["on"] != iv["on"].shift()).cumsum()
  return (
      iv.groupby(grp)
      .agg(start=("start", "min"), end=("end", "max"), on=("on", "first"))
      .reset_index(drop=True)
  )


def operation_intervals(
    beam,
    ts_col="timestamp",
    cols=("her_current", "ler_current"),
    on_thr=50.0,
    grid="1min",
    max_hold="2h",
    min_on="5min",
    shutdown_min="12h",
    rampup="15min",
    verbose=True,
):
  """Interval table (start, end, state): gapless and non-overlapping."""
  d = beam[[ts_col, *cols]].dropna()
  cur = (
      pd.Series(
          d[list(cols)].max(axis=1).to_numpy(),
          index=pd.DatetimeIndex(d[ts_col]),
      )
      .sort_index()
  )
  cur = cur.groupby(level=0).max()
  step = cur.index.to_series().diff().median()

  g = cur.resample(grid).max()
  g = g.ffill(
      limit=max(1, int(pd.Timedelta(max_hold) / pd.Timedelta(grid)))
  )
  on = (g > on_thr).to_numpy()

  change = np.r_[True, on[1:] != on[:-1]]
  iv = (
      pd.DataFrame({"start": g.index, "on": on, "run": np.cumsum(change)})
      .groupby("run")
      .agg(start=("start", "min"), on=("on", "first"))
  )
  iv["end"] = iv["start"].shift(-1)
  iv.loc[iv.index[-1], "end"] = g.index[-1] + pd.Timedelta(grid)
  iv = _merge(iv[["start", "end", "on"]].reset_index(drop=True))

  short_on = iv["on"] & ((iv["end"] - iv["start"]) < pd.Timedelta(min_on))
  iv.loc[short_on, "on"] = False
  iv = _merge(iv)

  dur = iv["end"] - iv["start"]
  iv["state"] = np.where(
      iv["on"],
      "BEAM_ON",
      np.where(
          dur >= pd.Timedelta(shutdown_min), "SHUTDOWN", "BEAM_OFF"
      ),
  )

  out = []
  for i, r in iv.iterrows():
    after_off = i > 0 and not iv["on"].iloc[i - 1]
    if (
        r["state"] == "BEAM_ON"
        and after_off
        and (r["end"] - r["start"]) > pd.Timedelta(rampup)
    ):
      mid = r["start"] + pd.Timedelta(rampup)
      out.append((r["start"], mid, "RAMP_UP"))
      out.append((mid, r["end"], "BEAM_ON"))
    elif r["state"] == "BEAM_ON" and after_off:
      out.append((r["start"], r["end"], "RAMP_UP"))
    else:
      out.append((r["start"], r["end"], r["state"]))
  res = pd.DataFrame(out, columns=["start", "end", "state"])

  if verbose:
    hrs = (
        ((res["end"] - res["start"]).dt.total_seconds() / 3600)
        .groupby(res["state"])
        .sum()
        .round(1)
    )
    print(
        f"rows: {len(cur)} | median row spacing: {step} | rows above threshold:"
        f" {(cur > on_thr).mean():.1%}"
    )
    print("hours per state:", hrs.to_dict())
    if step > pd.Timedelta("1min"):
      print(
          "WARNING: rows are more than 1 min apart. State boundaries are only"
          " as precise as the sampling, and windows like 5 min / 30 s cannot"
          " be resolved with such data."
      )
  return res


def get_off_intervals(iv):
  """Gibt eine Übersicht aller Zeiträume zurück, in denen der Strahl AUS war (BEAM_OFF & SHUTDOWN)."""
  off_df = iv[iv["state"].isin(["BEAM_OFF", "SHUTDOWN"])].copy()
  off_df["duration"] = off_df["end"] - off_df["start"]
  off_df["duration_hours"] = (
      off_df["duration"].dt.total_seconds() / 3600
  ).round(2)
  return off_df[["start", "end", "state", "duration_hours"]].reset_index(
      drop=True
  )


def label_states(df, iv, ts_col="timestamp"):
  """Adds the column 'state' to a block (use the same iv for every block)."""
  idx = pd.IntervalIndex.from_arrays(iv["start"], iv["end"], closed="left")
  pos = idx.get_indexer(pd.DatetimeIndex(df[ts_col]))
  out = df.copy()
  out["state"] = np.where(
      pos >= 0, iv["state"].to_numpy()[np.clip(pos, 0, len(iv) - 1)], "NO_DATA"
  )
  return out


def summary(iv, sbl_times):
  """Hours per state + the state each SBL was in (1 min before the event)."""
  hours = (
      ((iv["end"] - iv["start"]).dt.total_seconds() / 3600)
      .groupby(iv["state"])
      .agg(["sum", "count"])
  )
  hours.columns = ["hours", "intervals"]
  idx = pd.IntervalIndex.from_arrays(iv["start"], iv["end"], closed="left")
  sbl = pd.DatetimeIndex(pd.to_datetime(pd.Series(sbl_times)))
  pos = idx.get_indexer(sbl - pd.Timedelta("1min"))
  states = pd.Series(
      np.where(
          pos >= 0,
          iv["state"].to_numpy()[np.clip(pos, 0, len(iv) - 1)],
          "NO_DATA",
      )
  )
  hours["sbl_count"] = (
      states.value_counts().reindex(hours.index).fillna(0).astype(int)
  )
  return hours.round(1)


def plot_states(
    beam,
    iv,
    cols=("her_current", "ler_current"),
    ts_col="timestamp",
    print_off_details=True,
):
  """Beam current with shaded states and printed OFF/SHUTDOWN periods."""
  import matplotlib.pyplot as plt

  colors = {
      "SHUTDOWN": "#b0b0b0",
      "BEAM_OFF": "#f4c542",
      "RAMP_UP": "#8ecae6",
      "BEAM_ON": None,
  }

  fig, ax = plt.subplots(figsize=(14, 4))
  for c in cols:
    ax.plot(beam[ts_col], beam[c], lw=0.6, label=c)

  for _, r in iv.iterrows():
    if colors[r["state"]]:
      ax.axvspan(
          r["start"], r["end"], color=colors[r["state"]], alpha=0.35, lw=0
      )

  ax.set_ylabel("Beam current")
  ax.legend(loc="upper left")
  ax.set_title(
      "Gray = SHUTDOWN, yellow = BEAM_OFF, blue = RAMP_UP, white = BEAM_ON"
  )
  plt.tight_layout()
  plt.show()

  if print_off_details:
    print("\n" + "=" * 60)
    print(" DETAILS DER INAKTIVITÄTS-PHASEN (BEAM OFF / SHUTDOWN)")
    print("=" * 60)
    off_df = get_off_intervals(iv)
    for _, row in off_df.iterrows():
      print(
          f"[{row['state']:8s}] Von: {row['start']}  Bis: {row['end']} "
          f"| Dauer: {row['duration_hours']:6.2f} Std."
      )
    print("=" * 60 + "\n")