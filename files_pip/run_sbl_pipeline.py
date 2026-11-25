"""
run_sbl_pipeline.py - Runs the 5-step SBL anomaly-detection pipeline end to end.

    Step 1  pipeline_preprocess.py   time zones, BEAM_ON filter, per-block cleaning (no raw merge)
    Step 2  pipeline_features.py     backward rolling-window features per block
    Step 3  pipeline_models.py       z-score baseline + Isolation Forest, one score per row
    Step 4  pipeline_models.py       chronological split, training on normal rows only,
                                      score fusion across blocks
    Step 5  pipeline_validation.py   per-event detection rate, lead time, false alarms/hour,
                                      bootstrap CIs, chance-level reference

From a notebook:
    from run_sbl_pipeline import run_pipeline
    out = run_pipeline(
        {"beam": ds_beam, "background": ds_background, "injection": ds_injection,
         "vacuum": ds_vacuum, "position": ds_position},
        sbl_times,                                   # 1D Series/DatetimeIndex of the SBL timestamps
        {"block_tz": "Europe/Berlin", "sbl_tz": "Europe/Berlin"},   # only what you need to override
    )
    out["summary"]["iforest | ALL"]                  # table for one model / block
    plot_event_locked(out["fused"]["iforest"], out["sbl"], out["cfg"], out["split"])

From a terminal:
    python run_sbl_pipeline.py --dataset-dir Dataset --sbl-file Dataset/SBL_timestamps_Germany.csv
"""
import argparse

import numpy as np
import pandas as pd

from .pipeline_config import DEFAULTS
from .pipeline_features import build_features
from .pipeline_models import fuse_scores, make_split, score_block
from .pipeline_preprocess import preprocess
from .pipeline_validation import evaluate_series, injection_veto, plot_event_locked  # noqa: F401


def _section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def run_pipeline(blocks: dict, sbl_times, cfg_overrides: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **(cfg_overrides or {})}

    # ---------------- step 1 ----------------
    _section("Step 1: preprocessing")
    pre = preprocess(blocks, sbl_times, cfg)
    data, sbl = pre["data"], pre["sbl"]
    print(pre["blocks_report"].to_string(index=False))
    print(pre["events_report"].to_string(index=False))
    spacings = pre["blocks_report"].set_index("block")["median_spacing_s"]
    if (spacings > 60).any():
        print("\nNOTE: blocks with rows more than 1 minute apart: "
              f"{spacings[spacings > 60].to_dict()} - effects shorter than a few row spacings "
              "cannot be resolved there; windows were widened automatically.")

    # ---------------- step 2 ----------------
    _section("Step 2: feature engineering")
    feats = {n: build_features(d, cfg) for n, d in data.items()}
    for n, F in feats.items():
        print(f"  {n}: {F.shape[1]} features, window {F.attrs['window_s'] / 60:.1f} min "
              f"(median row spacing {F.attrs['spacing_s']:.0f} s)")

    # ---------------- steps 3 + 4 ----------------
    _section("Steps 3+4: training on normal rows only, chronological split")
    split = make_split(data, cfg)
    print(f"  train until {split['train_end']} | calibration until {split['calib_end']} | test after")
    grid = cfg["grid"]
    if grid == "auto":
        grid = f"{max(30, int(np.ceil(spacings.min() / 10) * 10))}s"
    print(f"  common score grid: {grid}")

    scores = {m: {} for m in cfg["methods"]}
    normed, fused = {}, {}
    for m in cfg["methods"]:
        for n, F in feats.items():
            try:
                scores[m][n] = score_block(F, sbl, cfg, split, m)
            except ValueError as e:
                print(f"  [{m}] block '{n}' skipped: {e}")
        normed[m] = fuse_scores(scores[m], sbl, cfg, split, grid)
        fused[m] = normed[m].max(axis=1)

    # ---------------- step 5 ----------------
    _section("Step 5: validation on the held-out test period")
    veto = None
    if cfg["use_injection_veto"]:
        any_idx = next(iter(fused.values())).index
        veto = injection_veto(pre["inj_active"], any_idx, grid, cfg)

    summary, per_event = {}, {}
    for m in cfg["methods"]:
        for label, series in [("ALL", fused[m])] + [(n, normed[m][n]) for n in normed[m].columns]:
            key = f"{m} | {label}"
            try:
                summary[key], per_event[key] = evaluate_series(series, sbl, cfg, split, veto)
            except ValueError as e:
                print(f"  {key}: not evaluated - {e}")
    for key, table in summary.items():
        print(f"\n--- {key} ---")
        print(table.to_string(index=False))
    print("\nHow to read: det_rate must clearly exceed chance_det_rate, and det_ci95 shows how "
          "uncertain it is with this few test events. median_lead_s = seconds before the SBL of "
          "the first alarm; fa_per_h = false-alarm episodes per hour of normal test time.")

    return dict(cfg=cfg, pre=pre, features=feats, split=split, grid=grid, scores=scores,
                normed=normed, fused=fused, summary=summary, per_event=per_event, sbl=sbl)


def _parse_args():
    p = argparse.ArgumentParser(description="Run the SBL anomaly-detection pipeline.")
    p.add_argument("--dataset-dir", default="Dataset")
    p.add_argument("--sbl-file", default="Dataset/SBL_timestamps_Germany.csv")
    p.add_argument("--block-tz", default=None, help='e.g. "Europe/Berlin" if timestamps are local')
    p.add_argument("--methods", nargs="+", default=list(DEFAULTS["methods"]), choices=["zscore", "iforest"])
    return p.parse_args()


if __name__ == "__main__":
    from data_loading import load_all_blocks, load_sbl_timestamps

    a = _parse_args()
    blocks = load_all_blocks(a.dataset_dir)
    sbl_df = load_sbl_timestamps(a.sbl_file)
    run_pipeline(blocks, sbl_df["timestamp"],
                 {"block_tz": a.block_tz, "sbl_tz": a.block_tz, "methods": tuple(a.methods)})
