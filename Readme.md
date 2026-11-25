# Sudden beam loss precursors at SuperKEKB

Code of the master's thesis *Predictive Analysis of Sudden Beam Loss Events – A Statistical and Machine-Learning
Study* (Justus-Liebig-Universität Gießen, II. Physikalisches Institut, 2026).

The analysis uses five slow-control data blocks (beam, background, injection, vacuum, position; 44 features)
and a list of 128 SBL times between November 2025 and May 2026. It asks whether any channel changes, or behaves
unusually, before an SBL:

1. change frequency and last change before each SBL, compared with chance and with random times,
2. distribution of the changes (ECDF, Kolmogorov–Smirnov test),
3. unsupervised anomaly detection (Isolation Forest, one model per block),
4. supervised risk model ("does an SBL follow within 2 h?", logistic regression and XGBoost).

**Main result:** the slow-control data describe the conditions under which SBLs are more frequent (above all a
high beam current), but they do not announce individual SBLs. Most channels are recorded as 2-h snapshots,
which is far too coarse for losses that develop within about a millisecond.

## Repository structure

| Path | Content |
|---|---|
| `00_main_sbl_analysis.ipynb` | One notebook that runs all analyses in the order of the thesis chapters |
| `files/` | Python modules with the analysis code (the notebook only calls them) |
| `files/snapshot_analysis/` | Step-by-step scripts of the snapshot analysis (`s0.py` … `s8.py`) |
| `files_pip/` | Earlier block-wise anomaly-detection pipeline (not used for the final results) |

### Modules used in the thesis

| Module | Content | Thesis |
|---|---|---|
| `data_loading.py` | Read the CSV blocks, rename the EPICS channels to readable names, load the SBL times | Ch. 3 |
| `remove_shutdown.py` | Remove given time ranges (winter shutdown) from every block | Ch. 3 |
| `operation_state.py` | Beam state (BEAM_ON / BEAM_OFF / SHUTDOWN) from the HER and LER currents | Ch. 3 |
| `beam_state_analysis.py` | Beam state of each ring at every SBL, rows per state | Ch. 3 |
| `eda_plots.py` | Overview plots: time series with SBL lines, distributions, events per day | Ch. 3 |
| `snapshot_analysis.py` | Sampling structure, clock check, raw (unfilled) blocks, reporting activity | Ch. 3, 5 |
| `p_before_sbl.py` | Change frequency P in windows before the SBLs | Ch. 5 |
| `enrichment_analysis.py` | Rate inside vs. outside the SBL windows, with buffer before the SBL | Ch. 5 |
| `plot_buffer.py` | Figure: enrichment vs. buffer | Ch. 5 |
| `last_change_before_sbl.py` | Last row and last changed feature before each SBL | Ch. 5 |
| `last_change_vs_control.py` | Last change before an SBL compared with ordinary times | Ch. 5 |
| `ks_overview.py`, `ecdf_plot.py` | KS distance per feature and window, ECDF figures | Ch. 5 |
| `sbl_iforest.py` | Isolation Forest per block, chance levels, feature counts, heatmaps | Ch. 5 |
| `risk_model.py` | Supervised risk model on the 2-h snapshots, ROC/PR, feature importance | Ch. 5 |
| `discussion_plots.py` | Snapshot age vs. precursor time scale, detectable rate | Ch. 6 |

The other modules in `files/` (e.g. `sbl_pipeline.py`, `pip_v01.py`, `sbl_anomaly.py`, `sbl_labeling.py`,
`diff_distribution_before_sbl.py`, `feature_scores.py`, `sbl_vs_control.py`) are earlier or exploratory versions.
They are kept for reference and are not needed to reproduce the thesis.

## Usage

1. Install Python 3.12 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`
   and `jupyter`. Versions used in the thesis: Python 3.12.10, NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.2,
   scikit-learn 1.7.2, XGBoost 2.1.1, Matplotlib 3.10.1.
2. Put the data files into a folder `Dataset/` in the repository root (see below).
3. Open `00_main_sbl_analysis.ipynb` and run it from top to bottom.

Expected files in `Dataset/`:

- forward-filled blocks: `01_ms_background.csv`, `02_ms_beam.csv`, `03_ms_position.csv`, `04_ms_injection.csv`,
  `05_ms_vacuum.csv`
- raw blocks (a value only when the channel reported): `11_BACKGROUND.csv`, `12_BEAM.csv`, `13_POSITION.csv`,
  `14_INJECTION.csv`, `15_VACUUM.csv`
- SBL list: `SBL_timestamps_Germany.csv` (one column, no header, German local time)

Notebook outputs are removed automatically on commit (`.gitattributes`, filter `nbstrip`), so that no data
values end up in the repository.

## Data availability

The slow-control data and the SBL list come from SuperKEKB/Belle II (KEK) and are **not public**.
They are therefore not part of this repository. Requests for access can be addressed to:

**Prof. Dr. Sören Lange**
II. Physikalisches Institut, Justus-Liebig-Universität Gießen
Web: [AG Lange](https://www.uni-giessen.de/de/fbz/fb07/fachgebiete/physik/institute/iipi/arbeitsgruppen/ag-lange)

## Author

Anas Shbat – master's thesis in physics, Justus-Liebig-Universität Gießen, 2026.
