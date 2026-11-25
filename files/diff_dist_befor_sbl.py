import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def add_diff_columns(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Add a '<feature>_diff' column (row-to-row change) for every feature. Returns a copy."""
    df = df.copy()
    for col in feature_cols:
        df[f"{col}_diff"] = df[col].diff()
    return df


def time_before_next_sbl(df: pd.DataFrame, ts_col: str, sbl_times) -> np.ndarray:
    """Seconds until the next upcoming SBL for every row (NaN-safe, always >= 0)."""
    ts = df[ts_col].to_numpy(dtype="datetime64[ns]").astype("int64")
    sbl = np.sort(pd.to_datetime(pd.Series(sbl_times)).to_numpy(dtype="datetime64[ns]").astype("int64"))
    idx = np.searchsorted(sbl, ts)
    idx = np.clip(idx, 0, len(sbl) - 1)
    return (sbl[idx] - ts) / 1e9


def plot_diff_distributions(df: pd.DataFrame, ts_col: str, feature_cols, sbl_times, 
                            time_before="5min", bins=60, use_abs=True, log_y=True, 
                            df_name="BEAM_clean") -> pd.DataFrame:
    """
    Histogramm-Vergleich der Feature-Änderungen vor SBLs sowie ein finaler
    Übersichtsplot mit Ratio of Means und KS-Test p-Werten.
    """
    df = add_diff_columns(df, feature_cols)
    seconds_to_next_sbl = time_before_next_sbl(df, ts_col, sbl_times)
    before_window_sec = pd.Timedelta(time_before).total_seconds()
    in_window = seconds_to_next_sbl <= before_window_sec

    n = len(feature_cols)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7), squeeze=False)

    rows = []
    for j, col in enumerate(feature_cols):
        diff_col = f"{col}_diff"
        all_vals = df[diff_col].dropna()
        window_vals = df.loc[in_window, diff_col].dropna()
        if use_abs:
            all_vals, window_vals = all_vals.abs(), window_vals.abs()

        lo, hi = all_vals.quantile(0.001), all_vals.quantile(0.999)
        bin_edges = np.linspace(lo, hi, bins)

        ax_top, ax_bot = axes[0, j], axes[1, j]
        ax_top.hist(all_vals, bins=bin_edges, color="steelblue")
        ax_top.set_title(f"[{df_name}] {col}\nall data (n={len(all_vals)})", fontsize=9)
        if log_y:
            ax_top.set_yscale("log")

        ax_bot.hist(window_vals, bins=bin_edges, color="darkorange")
        ax_bot.set_title(f"[{df_name}] {col}\nlast {time_before} before SBL (n={len(window_vals)})", fontsize=9)
        ax_bot.set_xlabel("|diff|" if use_abs else "diff")
        if log_y:
            ax_bot.set_yscale("log")

        if len(all_vals) > 10 and len(window_vals) > 10:
            stat, p = ks_2samp(window_vals, all_vals)
        else:
            stat, p = np.nan, np.nan
            
        rows.append(dict(
            feature=col, n_all=len(all_vals), n_window=len(window_vals),
            mean_all=all_vals.mean(), mean_window=window_vals.mean(),
            ratio_mean=window_vals.mean() / all_vals.mean() if all_vals.mean() else np.nan,
            ks_stat=stat, ks_pvalue=p
        ))

    plt.tight_layout()
    plt.show()

    result = pd.DataFrame(rows).sort_values("ks_pvalue")

    # --- Zusätzlicher Übersichtsplot am Ende ---
    fig_summary, ax_sum = plt.subplots(figsize=(max(8, 1.5 * n), 5))
    
    # Farbschema: Dunkelrot für signifikant erhöht (>1, p<0.05), Dunkelblau für signifikant verringert (<1, p<0.05), Grau für nicht signifikant
    colors = []
    for _, r in result.iterrows():
        if r["ks_pvalue"] < 0.05:
            colors.append("crimson" if r["ratio_mean"] > 1 else "dodgerblue")
        else:
            colors.append("lightgray")

    bars = ax_sum.bar(result["feature"], result["ratio_mean"], color=colors, edgecolor="black", alpha=0.8)
    ax_sum.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Normal level (Ratio = 1.0)")
    
    # Text-Annotationen über den Balken (p-Werte)
    for bar, (_, r) in zip(bars, result.iterrows()):
        yval = bar.get_height()
        p_str = f"p={r['ks_pvalue']:.1e}" if pd.notna(r['ks_pvalue']) else "N/A"
        ax_sum.text(
            bar.get_x() + bar.get_width() / 2.0, yval + (0.02 * max(result["ratio_mean"])),
            p_str, ha="center", va="bottom", fontsize=8, rotation=0
        )

    ax_sum.set_ylabel("Mean Diff Ratio (Window / Overall)")
    ax_sum.set_title(f"[{df_name}] Feature Change Ratio ({time_before} pre-SBL vs. Overall) & KS-Test Significance", fontsize=11)
    ax_sum.grid(axis="y", linestyle=":", alpha=0.6)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    print(result.to_string(index=False))
    return result