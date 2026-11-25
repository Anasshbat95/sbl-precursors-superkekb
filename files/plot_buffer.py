import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_enrichment_vs_buffer(table, features=None, top=10, save_path=None):
    """table = output of enrichment_vs_buffer. One line per feature: enrichment vs. seconds left out.
    Falls to ~1 after a few seconds -> reaction to the beam loss. Stays high -> possible precursor."""
    t = table.copy()
    t.columns = [int(str(c).split()[1].rstrip("s")) for c in t.columns]
    if features is None:
        features = t.iloc[:, 0].sort_values(ascending=False).head(top).index
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    cmap = plt.get_cmap("tab10")
    for k, f in enumerate(features):
        y = t.loc[f].to_numpy(float)
        ax.plot(t.columns, np.where(y > 0, y, 0.05), "o-", color=cmap(k % 10), label=f)
    ax.axhline(1, color="black", ls="--", lw=1, label="1 = as often as normal")
    ax.set_yscale("log")
    ax.set_xlabel("seconds left out right before the SBL (buffer)")
    ax.set_ylabel("enrichment (rate in window / rate outside)")
    ax.set_title("Enrichment vs. buffer: does the effect survive without the last seconds?  (0 drawn at 0.05)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.3, which="both")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_changes_per_sbl(per_sbl, features, save_path=None):
    """per_sbl = first output of changes_per_sbl. Left: changes per SBL over time (one dot = one SBL).
    Right: how many SBLs have 0, 1, 2, ... changes. Many SBLs with >= 1 = general pattern;
    a few SBLs with many changes = single events."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained", gridspec_kw=dict(width_ratios=[2, 1]))
    cmap = plt.get_cmap("tab10")
    for k, f in enumerate(features):
        a1.plot(per_sbl["sbl_time"], per_sbl[f] + 0.08 * k, "o", ms=4, color=cmap(k), label=f)
        vc = per_sbl[f].value_counts().sort_index()
        a2.bar(vc.index + (k - (len(features) - 1) / 2) * 0.8 / len(features), vc.values,
               width=0.8 / len(features), color=cmap(k), label=f)
    a1.set_ylabel("changes in [SBL - window, SBL - buffer)")
    a1.set_title("changes before each SBL")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    a2.set_xlabel("changes before the SBL")
    a2.set_ylabel("number of SBLs")
    a2.set_title("how many SBLs have 0, 1, 2 ... changes")
    for f in features:
        n = int((per_sbl[f] > 0).sum())
        a2.text(0.98, 0.95 - 0.07 * features.index(f), f"{f}: {n}/{len(per_sbl)} SBLs with >= 1",
                transform=a2.transAxes, ha="right", fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
