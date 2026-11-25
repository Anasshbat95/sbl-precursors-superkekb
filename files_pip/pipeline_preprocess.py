"""
pipeline_preprocess.py - Step 1 of the SBL pipeline: data fusion & preprocessing.

The blocks are NOT merged row by row (different sampling rates would create NaNs or
duplicated values). Each block is cleaned on its own and keeps its own time axis; only
the anomaly SCORES are fused later (pipeline_validation.fuse_scores).

What happens here:
  * all timestamps -> UTC without tz info (blocks and SBL list on the same clock)
  * optional: keep only BEAM_ON rows (operation_state.py), so shutdown / no-beam periods
    cannot be learned as "normal"
  * per block: numeric columns only, short forward-fill, drop rows still incomplete,
    drop constant columns
  * optional: log10 for columns that span several orders of magnitude (pressure, luminosity)
  * report: sampling rate per block, rows kept, SBL clustering, how many SBLs lie in BEAM_ON
"""
import numpy as np
import pandas as pd

from operation_state import label_states, operation_intervals
from files.sbl_anomaly import to_utc_naive


def _convert_timestamps(df, cfg):
    out = df.copy()
    ts = to_utc_naive(df[cfg["ts_col"]], cfg["block_tz"])
    out[cfg["ts_col"]] = ts.astype("datetime64[ns]").to_numpy()      # one resolution everywhere
    return out.dropna(subset=[cfg["ts_col"]])


def _log_columns(d, report_rows):
    """log10(x + eps) for non-negative columns whose 1%..99% range of positive values spans > 1e3."""
    transformed = []
    for c in d.columns:
        v = d[c]
        if v.min() < 0:
            continue
        pos = v[v > 0]
        if len(pos) < 50:
            continue
        lo, hi = pos.quantile(0.01), pos.quantile(0.99)
        if lo > 0 and hi / lo > 1e3:
            d[c] = np.log10(v + lo * 0.1)
            transformed.append(c)
    report_rows["log10_columns"] = transformed
    return d


def _clean_block(df, cfg):
    d = df.drop(columns=cfg["ts_col"]).select_dtypes("number").reset_index(drop=True)
    d.index = pd.DatetimeIndex(df[cfg["ts_col"]].to_numpy())
    d = d[d.index.notna()].sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d.ffill(limit=cfg["ffill_limit"]).dropna()
    return d.loc[:, d.std() > 0]


def preprocess(blocks: dict, sbl_times, cfg: dict) -> dict:
    """Returns dict with:
        data    {block: DataFrame (DatetimeIndex, numeric features)}
        sbl     sorted DatetimeIndex of all SBL events (UTC, tz-naive)
        iv      operation-state interval table (or None)
        inj_active  boolean Series (or None), True where the injection rate is above 0
        blocks_report, events_report  DataFrames for printing
    """
    ts_col = cfg["ts_col"]
    sbl = pd.DatetimeIndex(to_utc_naive(
        sbl_times.iloc[:, 0] if isinstance(sbl_times, pd.DataFrame) else sbl_times, cfg["sbl_tz"]
    ).dropna().sort_values()).astype("datetime64[ns]")

    conv = {n: _convert_timestamps(df, cfg) for n, df in blocks.items()}

    # --- keep BEAM_ON only -------------------------------------------------------------
    iv = None
    beam = conv.get(cfg["beam_block"])
    if cfg["use_operation_filter"]:
        if beam is not None and all(c in beam.columns for c in cfg["current_cols"]):
            iv = operation_intervals(beam, ts_col=ts_col, cols=cfg["current_cols"],
                                     on_thr=cfg["on_thr"], verbose=False)
        else:
            print(f"NOTE: operation filter skipped - block '{cfg['beam_block']}' with columns "
                  f"{cfg['current_cols']} not found.")

    data, rows_before, log_info = {}, {}, {}
    inj_active = None
    for name, df in conv.items():
        rows_before[name] = len(df)
        if iv is not None:
            df = label_states(df, iv, ts_col)
            df = df[df["state"] == "BEAM_ON"].drop(columns="state")
        d = _clean_block(df, cfg)

        if name == cfg["injection_block"]:                     # keep injection activity BEFORE any log transform
            cols = cfg["injection_rate_cols"] or [c for c in d.columns if "inj_rate" in c]
            if cols:
                inj_active = (d[cols] > 0).any(axis=1)
        info = {}
        if cfg["auto_log"]:
            d = _log_columns(d, info)
        log_info[name] = info.get("log10_columns", [])
        data[name] = d

    # --- reports -----------------------------------------------------------------------
    rows = []
    for name, d in data.items():
        spacing = d.index.to_series().diff().dt.total_seconds().median()
        rows.append(dict(block=name, rows_in=rows_before[name], rows_kept=len(d),
                         kept_pct=round(100 * len(d) / max(rows_before[name], 1), 1),
                         features=d.shape[1], median_spacing_s=round(float(spacing), 1),
                         log10_columns=len(log_info[name])))
    blocks_report = pd.DataFrame(rows)

    gaps = pd.Series(sbl).diff().dt.total_seconds().dropna()
    margin = pd.Timedelta(cfg["normal_margin"]).total_seconds()
    ev = dict(n_events=len(sbl),
              min_gap_s=float(gaps.min()) if len(gaps) else np.nan,
              median_gap_min=round(float(gaps.median()) / 60, 1) if len(gaps) else np.nan,
              sbl_pairs_closer_than_margin=int((gaps < margin).sum()))
    if iv is not None:
        idx = pd.IntervalIndex.from_arrays(iv["start"], iv["end"], closed="left")
        pos = idx.get_indexer(sbl - pd.Timedelta("1min"))
        st = np.where(pos >= 0, iv["state"].to_numpy()[np.clip(pos, 0, len(iv) - 1)], "NO_DATA")
        ev["events_in_beam_on"] = int((st == "BEAM_ON").sum())
    if inj_active is not None:
        ev["injection_active_fraction"] = round(float(inj_active.mean()), 3)
    events_report = pd.DataFrame([ev])

    return dict(data=data, sbl=sbl, iv=iv, inj_active=inj_active,
                blocks_report=blocks_report, events_report=events_report)
