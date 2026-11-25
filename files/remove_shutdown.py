"""
remove_shutdown.py - Standalone script: you give the shutdown time ranges yourself
(as start/end pairs), and it removes every row of a block that falls inside any
of them. No beam-current detection, no dependency on operation_state.py.

Usage:
    from remove_shutdown import remove_time_ranges

    shutdown_periods = [
        ("2025-12-20", "2026-01-25"),   # winter shutdown
        ("2026-03-01 10:00", "2026-03-01 14:00"),   # a maintenance day
    ]

    BEAM_clean   = remove_time_ranges(BEAM_clean, shutdown_periods)
    VACUUM_clean = remove_time_ranges(VACUUM_clean, shutdown_periods)   # same list for every block

You can also run this file directly to clean CSV files from the command line:
    python remove_shutdown.py data.csv "2025-12-20,2026-01-25" "2026-03-01 10:00,2026-03-01 14:00"
"""
import sys

import pandas as pd


def remove_time_ranges(df: pd.DataFrame, ranges, ts_col: str = "timestamp") -> pd.DataFrame:
    """Remove every row whose timestamp falls inside any [start, end) range.

    ranges: list of (start, end) pairs, each parseable by pandas.to_datetime,
            e.g. [("2025-12-20", "2026-01-25"), ("2026-03-01 10:00", "2026-03-01 14:00")].
    """
    ts = pd.to_datetime(df[ts_col])
    drop = pd.Series(False, index=df.index)
    for start, end in ranges:
        start, end = pd.to_datetime(start), pd.to_datetime(end)
        drop |= (ts >= start) & (ts < end)
    removed = int(drop.sum())
    print(f"Removed {removed} of {len(df)} rows ({removed / max(len(df), 1):.1%}) inside {len(ranges)} range(s).")
    return df.loc[~drop].reset_index(drop=True)


def _parse_range_arg(arg: str):
    start, end = arg.split(",")
    return start.strip(), end.strip()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python remove_shutdown.py <csv_file> \"start1,end1\" [\"start2,end2\" ...]")
        sys.exit(1)

    csv_path = sys.argv[1]
    ranges = [_parse_range_arg(a) for a in sys.argv[2:]]

    df = pd.read_csv(csv_path)
    cleaned = remove_time_ranges(df, ranges)

    out_path = csv_path.rsplit(".", 1)[0] + "_no_shutdown.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
