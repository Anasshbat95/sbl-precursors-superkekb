"""
merge_utils.py - Row-level combination of the 5 blocks with pd.merge_asof.

NOTE: this duplicates low-frequency blocks' values across many rows of the
high-frequency blocks (see the earlier discussion in chat). It is kept here
only as a reference / quality-control tool (e.g. to check how far apart the
blocks' timestamps really are), NOT as the basis for model training. For
training, keep the blocks separate (see sbl_labeling.py, sbl_anomaly.py).

Usage:
    from merge_utils import merge_asof_nearest

    combined = merge_asof_nearest(sbl_times, blocks)
    print(combined.filter(like="TimeDiff_sec").describe())
"""
import pandas as pd


def merge_asof_nearest(anchor: pd.DataFrame, blocks: dict[str, pd.DataFrame], ts_col: str = "timestamp",
                       direction: str = "backward") -> pd.DataFrame:
    """Attach the most recent (or nearest) row of every block to each row of `anchor` (e.g. the SBL events).

    Adds a 'TimeDiff_sec_<name>' column per block so you can check the match quality afterwards.
    """
    combined = anchor.copy().sort_values(ts_col).reset_index(drop=True)
    for name, df in blocks.items():
        sensor = df.copy().sort_values(ts_col).reset_index(drop=True)
        sensor[f"timestamp_{name}"] = sensor[ts_col]
        combined = pd.merge_asof(combined, sensor, on=ts_col, direction=direction, suffixes=("", f"_{name}"))
        combined[f"TimeDiff_sec_{name}"] = (combined[ts_col] - combined[f"timestamp_{name}"]).dt.total_seconds().abs()
    return combined
