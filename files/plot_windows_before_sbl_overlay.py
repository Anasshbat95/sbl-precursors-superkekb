"""
plot_windows_before_sbl_overlay.py - Overlay-Modus für alle 128 SBL-Events.
Plots all SBL events overlaid on a shared relative time axis (t = 0 is SBL).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from files.anomaly_detection_plot import _make_model


def _as_sbl_series(sbl_times) -> pd.Series:
    """Konvertiert sbl_times verlässlich in eine sortierte Pandas Series."""
    if isinstance(sbl_times, pd.DataFrame):
        if sbl_times.shape[1] != 1:
            raise ValueError(
                f"sbl_times ist ein DataFrame mit {sbl_times.shape[1]} Spalten - genau 1 erwartet."
            )
        sbl_times = sbl_times.iloc[:, 0]
    return pd.to_datetime(pd.Series(sbl_times).reset_index(drop=True)).sort_values().reset_index(drop=True)


def plot_feature_windows_before_sbl_overlay(df: pd.DataFrame, ts_col: str, feature_cols, sbl_times,
                                            window="5min", sbl_indices=None,
                                            df_name="data", marker_size=4, line_alpha=0.2, point_alpha=0.4,
                                            show_anomalies=True, anomaly_feature_cols=None,
                                            method="isolation_forest", contamination=0.05) -> pd.DataFrame:
    """Schneidet das Zeitfenster vor jedem SBL aus und legt alle Events auf einer
    gemeinsamen relativen Zeitachse übereinander (0 = SBL-Moment).
    """
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    sbl_sorted = _as_sbl_series(sbl_times)
    
    # Standardmäßig ALLE 128 SBL-Events nehmen
    if sbl_indices is None:
        sbl_indices = list(range(len(sbl_sorted)))
    else:
        sbl_indices = [i for i in sbl_indices if 0 <= i < len(sbl_sorted)]

    win_td = pd.Timedelta(window)
    win_sec = win_td.total_seconds()

    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d = d.sort_values(ts_col)

    kept_rows = []
    empty_indices = []

    for i in sbl_indices:
        sbl_t = sbl_sorted.iloc[i]
        w_start = sbl_t - win_td
        # Daten von (SBL - window) bis exakt SBL aufnehmen
        mask = (d[ts_col] >= w_start) & (d[ts_col] <= sbl_t)
        seg = d.loc[mask].copy()
        
        if seg.empty:
            empty_indices.append(i)
            continue

        # Relative Zeit in Sekunden: SBL-Moment ist exakt 0.0
        seg["rel_sec_to_sbl"] = (seg[ts_col] - sbl_t).dt.total_seconds()
        seg["sbl_index"] = i
        seg["sbl_time"] = sbl_t
        kept_rows.append(seg)

    if empty_indices:
        print(f"Hinweis: Für {len(empty_indices)} von {len(sbl_indices)} Events wurden im Fenster keine Daten gefunden.")

    if not kept_rows:
        raise ValueError("Keine Daten in den ausgewählten Fenstern gefunden!")

    kept = pd.concat(kept_rows, ignore_index=True)

    # Anomalienerkennung auf den ausgewählten Fenstern
    if show_anomalies:
        fit_cols = anomaly_feature_cols or feature_cols
        X = kept[fit_cols]
        valid = X.notna().all(axis=1)
        if valid.sum() < 20:
            print("Zu wenige valide Datenzeilen für Anomalienerkennung – wird übersprungen.")
            kept["is_anomaly"] = False
        else:
            Xs = StandardScaler().fit_transform(X[valid])
            model = _make_model(method, contamination, n_samples=valid.sum())
            pred = model.fit_predict(Xs) if method == "lof" else model.fit(Xs).predict(Xs)
            kept["is_anomaly"] = False
            kept.loc[X[valid].index, "is_anomaly"] = (pred == -1)

    # Erstellen des Overlay-Plots
    n = len(feature_cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), squeeze=False, sharex=True)
    axes = axes[:, 0]

    unique_events = kept["sbl_index"].unique()

    for ax, col in zip(axes, feature_cols):
        # Alle Events einzeln mit transparenter Farbe zeichnen
        for idx in unique_events:
            ev_data = kept[kept["sbl_index"] == idx]
            ax.plot(ev_data["rel_sec_to_sbl"], ev_data[col],
                    color="#377eb8", lw=0.7, alpha=line_alpha)
            ax.scatter(ev_data["rel_sec_to_sbl"], ev_data[col],
                       color="#377eb8", s=marker_size**2, alpha=point_alpha)

        # Anomalien hervorheben
        if show_anomalies and "is_anomaly" in kept.columns:
            anom = kept[kept["is_anomaly"]]
            ax.scatter(anom["rel_sec_to_sbl"], anom[col],
                       color="#ff7f00", s=25, zorder=5, label="Anomalie", alpha=0.8)

        # ROTE LINIE EXAKT AUF DEM SBL-MOMENT (t = 0)
        ax.axvline(0, color="red", linewidth=2.0, alpha=0.95, label="SBL-Moment (t=0)", zorder=10)
        
        # Gestrichelte graue Linie für den Start des Fensters
        ax.axvline(-win_sec, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        
        ax.set_ylabel(col, fontsize=9)
        ax.set_yscale('log' )  # Optional: logarithmische Y-Achse, falls sinnvoll
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Zeit relativ zu SBL [Sekunden] (0 = SBL-Moment)", fontsize=10)
    
    # X-Achse leicht über t=0 hinaus strecken, um Punkte direkt bei t=0 gut zu sehen
    padding = win_sec * 0.1
    axes[-1].set_xlim(-win_sec - padding, win_sec * 0.15)
    ax.set_yscale('log' )
    anomaly_note = f" - {method}, {int(kept['is_anomaly'].sum())} Anomalien" if show_anomalies else ""
    fig.suptitle(f"[{df_name}] Overlay von {len(unique_events)} SBL-Events (Fenster: {window} vor SBL){anomaly_note}",
                 y=1.01, fontsize=11)

    plt.tight_layout()
    plt.show()

    return kept