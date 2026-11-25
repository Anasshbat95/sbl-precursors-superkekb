import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

# Increase default resolution for all matplotlib figures
plt.rcParams["figure.dpi"] = 150       # resolution shown in notebook/inline
plt.rcParams["savefig.dpi"] = 300      # resolution used when saving to file


def _seconds_to_next_sbl(ts, sbl_times):
    """Seconds from each row to the next SBL at or after it; NaN after the last SBL."""
    t = pd.to_datetime(pd.Series(ts)).to_numpy(dtype="datetime64[ns]").astype("int64")
    s = np.sort(pd.to_datetime(pd.Series(sbl_times)).to_numpy(dtype="datetime64[ns]").astype("int64"))
    idx = np.searchsorted(s, t)
    out = np.full(len(t), np.nan)
    ok = idx < len(s)
    out[ok] = (s[idx[ok]] - t[ok]) / 1e9
    return out


def plot_all_features_single_plot(
    df, dataset_name, distance_threshold=300, scale_data=True, ax=None,
    show_peak_value=False, kind="kde", sbl_times=None, ts_col="timestamp"
):
    """kind="kde": density (as before). kind="ecdf": cumulative distribution function F(x).
    sbl_times given -> only rows BEFORE an SBL (0 .. distance_threshold s before it);
    sbl_times=None  -> rows within distance_threshold s before OR after (old behaviour)."""
    if sbl_times is not None:
        sec = _seconds_to_next_sbl(df[ts_col], sbl_times)
        df_filtered = df[sec <= distance_threshold]        # NaN (after the last SBL) -> False
        window_text = f"{distance_threshold}s before SBL"
    else:
        df_filtered = df[df["dist_to_sbl_sec"] <= distance_threshold]
        window_text = f"dist <= {distance_threshold}s"
    exclude_cols = ["target", "dist_to_sbl_sec", "sbl_index", "sbl_diff_sec", "min_abs_dist"]

    # Keep only numeric feature columns, drop excluded columns and NaNs
    df_numeric = (
        df_filtered.select_dtypes(include=["float64", "int64"])
        .drop(columns=exclude_cols, errors="ignore")
        .reset_index(drop=True)
        .dropna()
    )

    if df_numeric.empty:
        print(f"No suitable numeric features found in {dataset_name}.")
        if ax is not None:
            ax.set_visible(False)
        return

    # Optionally standardize (z-score) the data before plotting
    if scale_data:
        scaled_array = StandardScaler().fit_transform(df_numeric)
        df_plot = pd.DataFrame(scaled_array, columns=df_numeric.columns)
        x_label = "Z-Score"
    else:
        df_plot = df_numeric
        x_label = "Value"

    # --- Compute peak density height per feature (also used to sort the ECDF plot,
    #     so every feature has the SAME colour and legend position in both figures) ---
    peak_heights = {}
    for col in df_plot.columns:
        vals = df_plot[col].values
        if np.std(vals) > 0:
            x_grid = np.linspace(vals.min(), vals.max(), 200)
            peak_heights[col] = gaussian_kde(vals)(x_grid).max()
        else:
            peak_heights[col] = 0
    sorted_features = sorted(peak_heights, key=lambda c: peak_heights[c], reverse=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    color_cycle = cycle(sns.color_palette("tab20", n_colors=min(len(sorted_features), 20)))
    linestyles = ["-", "--", "-.", ":"]
    handles, labels = [], []

    for i, col in enumerate(sorted_features):
        color = next(color_cycle)
        linestyle = linestyles[i // 20 % len(linestyles)]
        if kind == "kde":
            if np.std(df_plot[col].values) > 0:
                sns.kdeplot(data=df_plot, x=col, ax=ax, color=color, linestyle=linestyle,
                            linewidth=1.0, common_norm=False)
                handle = ax.lines[-1]
            else:                                   # constant feature: no density, show as a vertical line
                handle = ax.axvline(df_plot[col].iloc[0], color=color, linestyle=linestyle, linewidth=1.0)
        elif kind == "ecdf":
            v = np.sort(df_plot[col].values)
            (handle,) = ax.step(np.r_[v[0], v], np.r_[0, np.arange(1, len(v) + 1) / len(v)],
                                where="post", color=color, linestyle=linestyle, linewidth=1.0)
        else:
            raise ValueError("kind must be 'kde' or 'ecdf'.")
        label = f"{col} ({peak_heights[col]:.2f})" if show_peak_value else col
        handles.append(handle)
        labels.append(label)

    if kind == "ecdf":
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("F(x)", fontsize=7)
    else:
        ax.set_ylim(bottom=0)                       # no upper limit: peaks above 1 stay visible
        ax.set_ylabel("Density", fontsize=7)
    ax.set_title(f"{dataset_name} ({window_text}, n={len(df_plot)})", fontsize=8, pad=4, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.legend(
        handles=handles, labels=labels, fontsize=5,
        title="Feature (sorted by peak height)", title_fontsize=6,
        framealpha=0.6, handlelength=1.2, handletextpad=0.3, labelspacing=0.15, borderpad=0.25,
        loc="upper right" if kind == "kde" else "lower right",
    )
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_distribution_figures(datasets, distance_threshold=300, sbl_times=None, save_dir=None,
                              scale_data=True, show_peak_value=True):
    """Two figures with the same layout: 1) densities, 2) cumulative distribution functions."""
    figs = {}
    for kind in ("kde", "ecdf"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.ravel()
        for ax, (name, df) in zip(axes, datasets.items()):
            plot_all_features_single_plot(df, name, distance_threshold=distance_threshold, scale_data=scale_data,
                                          ax=ax, show_peak_value=show_peak_value, kind=kind, sbl_times=sbl_times)
        for ax in axes[len(datasets):]:
            ax.set_visible(False)
        fig.suptitle("Density" if kind == "kde" else "Cumulative distribution function", fontsize=11)
        plt.tight_layout()
        if save_dir:                                # save BEFORE show, otherwise the file can be empty
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"feature_{kind}_{distance_threshold}s.png"), bbox_inches="tight")
        plt.show()
        figs[kind] = fig
    return figs


# ---------------- Main ----------------
if __name__ == "__main__":
    datasets = {
        "BEAM": beam_ds,                 # noqa: F821  (defined in your notebook)
        "BACKGROUND": background_ds,     # noqa: F821
        "INJECTION": injection_ds,       # noqa: F821
        "VACUUM": vacuum_ds,             # noqa: F821
        "POSITION": position_ds,         # noqa: F821
    }
    plot_distribution_figures(datasets, distance_threshold=300,
                              sbl_times=SBL_timestamps_Germany["timestamp"],   # noqa: F821
                              save_dir="Abbildung")
