"""
anomaly_detection_plot.py - Same principle as the scikit-learn anomaly-detection
gallery example (fit a model, color points by inlier/outlier), applied to a
time-series block instead of a 2D point cloud: each feature is drawn as a line,
with points the model flags as anomalous highlighted in orange, and the actual
SBL events marked as vertical dashed red lines (same style as your existing
multi-panel plot) so you can see whether the model's anomalies line up with the
real events.

Usage:
    from anomaly_detection_plot import plot_anomaly_detection

    feature_cols = [c for c in ds_beam.columns if c != "timestamp"]
    result = plot_anomaly_detection(
        ds_beam, "timestamp", feature_cols, sbl_times,
        start=None, end=None,              # default: complete range, like your current plot
        method="isolation_forest", contamination=0.05, df_name="beam",
    )
    # zoomed in on one period:
    plot_anomaly_detection(ds_beam, "timestamp", feature_cols, sbl_times,
                           start="2026-01-15", end="2026-02-15", df_name="beam")
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _as_sbl_series(sbl_times) -> pd.Series:
    """Accepts a Series, 1D array/list, or single-column DataFrame and always returns a
    1D datetime Series (handles the same (n,1)-DataFrame pitfall as diff_distribution_before_sbl.py)."""
    if isinstance(sbl_times, pd.DataFrame):
        if sbl_times.shape[1] != 1:
            raise ValueError(
                f"sbl_times is a DataFrame with {sbl_times.shape[1]} columns - expected exactly 1. "
                f"Pass a single column instead, e.g. sbl_times['{sbl_times.columns[0]}']."
            )
        sbl_times = sbl_times.iloc[:, 0]
    return pd.to_datetime(pd.Series(sbl_times).reset_index(drop=True))


def _make_model(method: str, contamination: float, n_samples: int = None):
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        return IsolationForest(contamination=contamination, random_state=42)
    if method == "lof":
        from sklearn.neighbors import LocalOutlierFactor
        return LocalOutlierFactor(n_neighbors=35, contamination=contamination)
    if method == "one_class_svm":
        from sklearn import svm
        return svm.OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
    if method == "one_class_svm_sgd":
        # same construction as the scikit-learn comparison gallery: a kernel approximation
        # (Nystroem) feeding a linear SGDOneClassSVM, so it scales to far more rows than a
        # plain OneClassSVM. n_components is capped to the available rows.
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.pipeline import make_pipeline
        n_components = min(150, (n_samples or 150) - 1)
        return make_pipeline(
            Nystroem(gamma=0.1, random_state=42, n_components=n_components),
            SGDOneClassSVM(nu=contamination, shuffle=True, fit_intercept=True,
                          random_state=42, tol=1e-6),
        )
    if method == "robust_covariance":
        from sklearn.covariance import EllipticEnvelope
        return EllipticEnvelope(contamination=contamination, random_state=42)
    raise ValueError(f"Unknown method '{method}'. Use 'isolation_forest', 'lof', "
                     "'one_class_svm', 'one_class_svm_sgd', or 'robust_covariance'.")


def plot_anomaly_detection(df: pd.DataFrame, ts_col: str, feature_cols, sbl_times,
                           start=None, end=None, method="isolation_forest", contamination=0.05,
                           df_name="data", log_y=True) -> pd.DataFrame:
    """Fit an unsupervised anomaly detector on `feature_cols`, then plot each feature as a time
    series (blue) with the rows the model flags as anomalous highlighted (orange), and the real
    SBL events as vertical dashed red lines.

    start / end: optional zoom window (anything pd.Timestamp understands, e.g. "2026-01-15").
                 Default None = the complete time range, like your current plot.
    method: "isolation_forest" (default, good general-purpose choice), "lof", "one_class_svm",
            "one_class_svm_sgd" (kernel-approximated, for larger row counts), or
            "robust_covariance" - the same 5 options as the scikit-learn comparison gallery.
    contamination: expected fraction of anomalous rows (0.05 = ~5%). This is a rough guess the
                   model uses to set its internal threshold, not a precise number you need to know.

    Returns the (zoomed) data with an added 'is_anomaly' column.
    """
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    if start is not None:
        d = d[d[ts_col] >= pd.Timestamp(start)]
    if end is not None:
        d = d[d[ts_col] <= pd.Timestamp(end)]
    d = d.sort_values(ts_col).reset_index(drop=True)
    if d.empty:
        raise ValueError(f"No rows left between start={start!r} and end={end!r}.")

    X = d[feature_cols]
    valid = X.notna().all(axis=1)
    X_valid = X[valid]
    if len(X_valid) < 20:
        raise ValueError(f"Only {len(X_valid)} rows have every feature non-NaN in this range - "
                         "too few to fit a model. Try a wider start/end window.")

    Xs = StandardScaler().fit_transform(X_valid)   # put all features on the same scale first

    model = _make_model(method, contamination, n_samples=len(Xs))
    pred = model.fit_predict(Xs) if method == "lof" else model.fit(Xs).predict(Xs)

    d["is_anomaly"] = False
    d.loc[X_valid.index, "is_anomaly"] = (pred == -1)

    sbl = _as_sbl_series(sbl_times)
    sbl = sbl[(sbl >= d[ts_col].min()) & (sbl <= d[ts_col].max())]

    n = len(feature_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.1 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, col in zip(axes, feature_cols):
        ax.plot(d[ts_col], d[col], color="#377eb8", lw=0.6, alpha=0.8)
        anom = d[d["is_anomaly"]]
        ax.scatter(anom[ts_col], anom[col], color="#ff7f00", s=10, zorder=5, label="predicted anomaly")
        for t in sbl:
            ax.axvline(t, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(col, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].plot([], [], color="red", linestyle="--", label="SBL event")  # legend entry only
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")

    time_range = f"{d[ts_col].min():%Y-%m-%d} to {d[ts_col].max():%Y-%m-%d}"
    n_anom = int(d["is_anomaly"].sum())
    fig.suptitle(f"[{df_name}] {method} - {time_range}\n"
                f"{n_anom}/{len(X_valid)} rows flagged anomalous (orange) - "
                f"{len(sbl)} SBL event(s) in range (red dashed)", y=1.0, fontsize=11)
    plt.tight_layout()
    plt.show()

    return d[[ts_col, "is_anomaly", *feature_cols]]
