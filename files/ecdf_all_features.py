"""
ecdf_all_features.py - Distribution functions (ECDF, KS distance D) for ALL features of ALL blocks.

    plot_ecdf_all_features(blocks, sbl)
        1  one figure per block: every feature, F_Fenster (window before SBL) vs. F_Normal
           (rows farther than normal_margin from every SBL), with the KS distance D
        2  one overview figure: D of every feature, sorted, coloured by block, one bar per window
        returns the table of all features (D, n, share of zeros, descriptive p-value)

Uses plot_ecdf_window_vs_all from diff_distribution_before_sbl.py, so the numbers are exactly the
same as in that function. Defaults: quantity="diff" (row-to-row change), reference="normal".

Usage:
    from ecdf_all_features import plot_ecdf_all_features
    table = plot_ecdf_all_features(blocks_on, sbl, windows=("5min", "30min"), save_dir="Abbildung/ecdf")
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from files.diff_distribution_before_sbl import plot_ecdf_window_vs_all

BLOCK_COLORS = {"beam": "#1f77b4", "background": "#ff7f0e", "injection": "#2ca02c",
                "vacuum": "#d62728", "position": "#9467bd"}


def plot_ecdf_all_features(blocks, sbl_times, windows=("5min", "30min"), quantity="diff", reference="normal",
                           normal_margin="15min", ts_col="timestamp", n_cols=4, dpi=150,
                           save_dir="Abbildung/ecdf", min_rows=10):
    windows = [windows] if isinstance(windows, str) else list(windows)
    tables = []
    for name, df in blocks.items():
        feats = [c for c in df.select_dtypes("number").columns if c != ts_col]
        if not feats:
            continue
        fig, tab = plot_ecdf_window_vs_all({name: df}, ts_col, feats, sbl_times, time_before=windows,
                                           quantity=quantity, reference=reference, normal_margin=normal_margin,
                                           n_cols=min(n_cols, len(feats)), dpi=dpi)
        fig.suptitle(f"{name}: distribution functions, window before SBL vs. "
                     f"{'normal operation' if reference == 'normal' else 'all rows'}  "
                     f"({'row-to-row change' if quantity == 'diff' else 'values'};  D = KS distance)", fontsize=11)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"ecdf_{name}_{quantity}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        tables.append(tab.assign(block=name))
    table = pd.concat(tables, ignore_index=True)
    if "D" not in table:
        table["D"] = np.nan
    table["enough_rows"] = table["n_window"] >= min_rows
    _plot_overview(table, windows, quantity, reference, save_dir, min_rows)
    cols = ["block", "feature", "window", "n_window", "n_all", "D", "frac_zero_window", "frac_zero_all",
            "median_window", "median_all", "p_value", "enough_rows"]
    table = table[[c for c in cols if c in table.columns]]
    return table.sort_values(["window", "D"], ascending=[True, False]).reset_index(drop=True)


def _plot_overview(table, windows, quantity, reference, save_dir, min_rows):
    piv = table.pivot_table(index=["block", "feature"], columns="window", values="D")
    piv = piv.reindex(columns=[w for w in windows if w in piv.columns])
    nwin = table.pivot_table(index=["block", "feature"], columns="window", values="n_window")
    order = piv.max(axis=1).sort_values().index
    piv, nwin = piv.loc[order], nwin.reindex(order)
    y = np.arange(len(piv))
    h = 0.8 / max(len(piv.columns), 1)
    fig, ax = plt.subplots(figsize=(10, 0.27 * len(piv) + 2), layout="constrained")
    for k, w in enumerate(piv.columns):
        colors = [BLOCK_COLORS.get(b, "gray") for b, _ in piv.index]
        alpha = 1.0 if k == len(piv.columns) - 1 else 0.45
        few = (nwin[w] < min_rows).to_numpy() if w in nwin else np.zeros(len(piv), bool)
        bars = ax.barh(y + (k - (len(piv.columns) - 1) / 2) * h, piv[w].fillna(0), h, color=colors, alpha=alpha,
                       edgecolor="black", lw=0.3)
        for b, f in zip(bars, few):
            if f:
                b.set_hatch("///")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{b}: {f}" for b, f in piv.index], fontsize=7)
    for lab, (b, _) in zip(ax.get_yticklabels(), piv.index):
        lab.set_color(BLOCK_COLORS.get(b, "black"))
    ax.set_xlim(0, 1)
    ax.set_xlabel("KS distance D  (0 = same distribution as normal, 1 = completely separated)")
    win_txt = ", ".join(f"{w} ({'light' if i < len(piv.columns) - 1 else 'dark'})" for i, w in enumerate(piv.columns))
    ax.set_title(f"All features: D between window before SBL and "
                 f"{'normal operation' if reference == 'normal' else 'all rows'}  "
                 f"({'row-to-row change' if quantity == 'diff' else 'values'})\n"
                 f"windows: {win_txt};  hatched = fewer than {min_rows} rows in the window", fontsize=10)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"ecdf_overview_D_{quantity}.png"), dpi=300, bbox_inches="tight")
    plt.show()
