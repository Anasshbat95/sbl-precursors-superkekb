"""
main.py - Runs the full SBL pipeline end to end:

    1. Load & rename the 5 raw blocks and the SBL timestamps.
    2. Determine BEAM_ON / RAMP_UP / BEAM_OFF / SHUTDOWN and keep only BEAM_ON.
    3. Label each block (risk / normal) with the SBL time windows.
    4. Exploratory plots (time series, feature distributions, missing values).
    5. Feature enrichment analysis (which feature reacts most to an SBL).
    6. Train a per-block RandomForest baseline (model_pipeline.py).
    7. Run the anomaly-detection pipeline across all blocks (sbl_anomaly.py).

Each step is a separate module, so you can also import and call these
functions individually from a notebook instead of running this script.

Usage:
    python main.py --dataset-dir Dataset --sbl-file Dataset/SBL_timestamps_Germany.csv
"""
import argparse

from data_loading import load_all_blocks, load_sbl_timestamps
from eda_plots import (
    plot_missing_value_overview,
    plot_sbl_events_per_day,
    plot_timeseries_multi_block,
)
from enrichment_analysis import (
    compute_enrichment_ratio,
    count_values_in_sbl_windows,
    plot_enrichment_ratio,
    plot_probability_per_feature,
)
from model_pipeline import build_pipeline, train_and_evaluate
from operation_state import label_states, operation_intervals, summary
from sbl_anomaly import run as run_anomaly_detection
from sbl_labeling import label_all_blocks


def parse_args():
    p = argparse.ArgumentParser(description="Run the SBL analysis pipeline.")
    p.add_argument("--dataset-dir", default="Dataset", help="Folder with the raw CSV files.")
    p.add_argument("--sbl-file", default="Dataset/SBL_timestamps_Germany.csv", help="CSV with SBL timestamps.")
    p.add_argument("--on-thr", type=float, default=50.0, help="Beam-current threshold for BEAM_ON.")
    p.add_argument("--skip-plots", action="store_true", help="Skip the exploratory plots (useful for CI/batch runs).")
    p.add_argument("--anomaly-model", choices=["linear", "lstm"], default="linear",
                   help="Backend for the anomaly-detection pipeline.")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Load
    print("Loading blocks...")
    blocks = load_all_blocks(args.dataset_dir)
    sbl_times = load_sbl_timestamps(args.sbl_file)
    for name, df in blocks.items():
        print(f"  {name}: {df.shape}")
    print(f"  SBL events: {len(sbl_times)}")

    # 2. Operation state: keep only times with beam
    print("\nDetermining operation state from BEAM_ON current...")
    iv = operation_intervals(blocks["beam"], cols=("her_current", "ler_current"), on_thr=args.on_thr)
    print(summary(iv, sbl_times["timestamp"]))
    blocks_on = {name: label_states(df, iv).query("state == 'BEAM_ON'").drop(columns="state")
                for name, df in blocks.items()}

    # 3. Label risk / normal per block
    print("\nLabeling blocks (risk / buffer / recovery / normal)...")
    labeled = label_all_blocks(blocks_on, sbl_times["timestamp"])
    for name, df in labeled.items():
        print(f"  {name}: {df['target'].value_counts().to_dict()}")

    # 4. EDA
    if not args.skip_plots:
        print("\nPlotting exploratory views...")
        plot_sbl_events_per_day(sbl_times["timestamp"])
        plot_missing_value_overview(blocks_on)
        plot_timeseries_multi_block(blocks_on, title_prefix="All blocks", sbl_datetimes=sbl_times["timestamp"])

    # 5. Enrichment analysis
    print("\nComputing feature enrichment near SBL events...")
    blocks_upper = {name.upper(): df for name, df in blocks_on.items()}
    _, _, p_in, group_of_feature = plot_probability_per_feature(sbl_times["timestamp"], blocks_upper,
                                                                 show_plot=not args.skip_plots)
    ratio, _ = compute_enrichment_ratio(sbl_times["timestamp"], blocks_upper)
    if not args.skip_plots:
        plot_enrichment_ratio(ratio, group_of_feature)

    # 6. Per-block supervised baseline (example: BEAM)
    print("\nTraining a per-block RandomForest baseline on BEAM...")
    pipeline = build_pipeline()
    train_and_evaluate(pipeline, labeled["beam"], plot=not args.skip_plots)

    # 7. Cross-block anomaly detection
    print("\nRunning the cross-block anomaly-detection pipeline...")
    run_anomaly_detection(blocks_on, sbl_times["timestamp"], {"model": args.anomaly_model})


if __name__ == "__main__":
    main()
