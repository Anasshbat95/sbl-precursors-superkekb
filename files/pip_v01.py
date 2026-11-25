"""Unsupervised, block-wise SBL analysis.

Run from the repository root with::

	python -m files.pip_v01

The five data blocks keep their own time axes. No rows are merged or interpolated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

try:
	from files.data_loading import load_all_blocks, load_sbl_timestamps
except ModuleNotFoundError:  # supports ``python pip_v01.py`` from files/
	from data_loading import load_all_blocks, load_sbl_timestamps


META_COLUMNS = {"nearest_sbl_index", "distance_to_sbl_s"}


def add_sbl_labels(df: pd.DataFrame, sbl_times, lookback="10min", buffer="30s") -> pd.DataFrame:
	"""Add nearest-SBL metadata without changing the original measurements."""
	out = df.copy()
	out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
	out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
	sbl = pd.to_datetime(pd.Series(sbl_times)).dropna().sort_values().to_numpy(dtype="datetime64[ns]")
	if len(sbl) == 0:
		raise ValueError("No valid SBL timestamps were supplied.")

	timestamps = out["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
	sbl_ns = sbl.astype("int64")
	pos = np.searchsorted(sbl_ns, timestamps)
	left = sbl_ns[np.clip(pos - 1, 0, len(sbl_ns) - 1)]
	right = sbl_ns[np.clip(pos, 0, len(sbl_ns) - 1)]
	left_distance = np.abs(timestamps - left)
	right_distance = np.abs(right - timestamps)
	nearest = np.where(left_distance <= right_distance, np.clip(pos - 1, 0, len(sbl_ns) - 1),
					   np.clip(pos, 0, len(sbl_ns) - 1))
	distance_s = np.minimum(left_distance, right_distance) / 1e9
	lookback_s = pd.Timedelta(lookback).total_seconds()
	buffer_s = pd.Timedelta(buffer).total_seconds()

	out["nearest_sbl_index"] = nearest
	out["distance_to_sbl_s"] = distance_s
	out["is_sbl_pre_window"] = (
		(distance_s <= lookback_s)
		& (timestamps < sbl_ns[nearest])
		& (distance_s >= buffer_s)
	)
	out["is_sbl_exclusion"] = distance_s < buffer_s
	return out


def robust_feature_scores(labeled: pd.DataFrame, lookback="10min") -> pd.DataFrame:
	"""Return one robust absolute z-score per feature and SBL event."""
	feature_cols = [c for c in labeled.select_dtypes("number").columns if c not in META_COLUMNS]
	values = labeled[feature_cols].apply(pd.to_numeric, errors="coerce")
	normal = ~labeled["is_sbl_pre_window"] & ~labeled["is_sbl_exclusion"]
	normal_values = values.loc[normal]
	median = normal_values.median()
	scale = (normal_values.quantile(0.75) - normal_values.quantile(0.25)) / 1.349
	scale = scale.where(scale > 0, normal_values.std()).replace(0, np.nan)
	z = ((values - median) / scale).abs()
	sbl_times = labeled.attrs["sbl_times"]
	rows = []
	for event, sbl_time in enumerate(sbl_times):
		window = labeled[(labeled["timestamp"] >= sbl_time - pd.Timedelta(lookback))
						 & (labeled["timestamp"] < sbl_time)]
		for feature in feature_cols:
			scores = z.loc[window.index, feature].dropna()
			rows.append({
				"event": event,
				"sbl_time": sbl_time,
				"feature": feature,
				"max_abs_z": scores.max() if len(scores) else np.nan,
				"n_values": int(len(scores)),
			})
	return pd.DataFrame(rows)


def train_block_models(labeled_blocks: dict[str, pd.DataFrame], random_state=42) -> dict:
	"""Fit one Isolation Forest per block on non-SBL normal rows."""
	models, scores = {}, {}
	for block, labeled in labeled_blocks.items():
		feature_cols = [c for c in labeled.select_dtypes("number").columns if c not in META_COLUMNS]
		values = labeled[feature_cols].apply(pd.to_numeric, errors="coerce")
		valid = values.notna().all(axis=1)
		normal = valid & ~labeled["is_sbl_pre_window"] & ~labeled["is_sbl_exclusion"]
		if int(normal.sum()) < 50:
			print(f"{block}: skipped ({int(normal.sum())} normal rows)")
			continue

		scaler = RobustScaler()
		X_train = scaler.fit_transform(values.loc[normal])
		X_all = scaler.transform(values.loc[valid])
		model = IsolationForest(n_estimators=300, contamination="auto", random_state=random_state)
		model.fit(X_train)

		result = labeled[["timestamp", "nearest_sbl_index", "distance_to_sbl_s",
						  "is_sbl_pre_window", "is_sbl_exclusion"]].copy()
		result["anomaly_score"] = np.nan
		result.loc[valid, "anomaly_score"] = -model.score_samples(X_all)
		result["block"] = block
		models[block] = {"model": model, "scaler": scaler, "features": feature_cols}
		scores[block] = result
		print(f"{block}: train={int(normal.sum())}, scored={int(valid.sum())}")
	return {"models": models, "scores": scores}


def plot_anomaly_scores(scores: dict[str, pd.DataFrame], sbl_times, block=None,
						sbl_indices=None, before=None, after=None, yscale=None):
	"""Plot anomaly scores with optional block, event, time, and y-scale filters.

	``before=None`` and ``after=None`` show the complete available time range.
	``sbl_indices=None`` draws all SBL lines; otherwise only selected indices are drawn.
	"""
	selected_scores = scores if block is None else {block: scores[block]}
	sbl = pd.to_datetime(pd.Series(sbl_times)).dropna().sort_values().reset_index(drop=True)
	selected_sbl = sbl if sbl_indices is None else sbl.iloc[list(sbl_indices)]
	start = pd.Timedelta(before) if before is not None else None
	end = pd.Timedelta(after) if after is not None else None

	fig, axes = plt.subplots(len(selected_scores), 1,
							 figsize=(16, 3 * len(selected_scores)),
							 sharex=True, squeeze=False)
	axes = axes[:, 0]
	for ax, (block_name, data) in zip(axes, selected_scores.items()):
		plot_data = data.dropna(subset=["timestamp", "anomaly_score"]).sort_values("timestamp")
		if start is not None or end is not None:
			left = selected_sbl.min() - start if start is not None else plot_data["timestamp"].min()
			right = selected_sbl.max() + end if end is not None else plot_data["timestamp"].max()
			plot_data = plot_data[(plot_data["timestamp"] >= left) & (plot_data["timestamp"] <= right)]
		ax.plot(plot_data["timestamp"], plot_data["anomaly_score"], color="steelblue", lw=0.8)
		for index, sbl_time in enumerate(selected_sbl):
			ax.axvline(sbl_time, color="firebrick", alpha=0.35, lw=0.8,
					   label="SBL" if index == 0 else None)
		if yscale is not None:
			ax.set_yscale(yscale)
		ax.set_ylabel(block_name)
		ax.grid(alpha=0.3)
		ax.legend(loc="upper right")
	axes[-1].set_xlabel("Zeit")
	fig.suptitle("Anomaly Scores pro Datenblock")
	plt.tight_layout()
	plt.show()
	return fig, axes


def plot_block_scores(scores: dict[str, pd.DataFrame], sbl_times, lookback="10min"):
	"""Plot one readable max-score panel per block and return the summary table."""
	rows = []
	sbl = pd.to_datetime(pd.Series(sbl_times)).dropna().sort_values()
	for event, sbl_time in enumerate(sbl):
		for block, data in scores.items():
			part = data[(data["timestamp"] >= sbl_time - pd.Timedelta(lookback))
						& (data["timestamp"] < sbl_time)]["anomaly_score"].dropna()
			rows.append({"event": event, "sbl_time": sbl_time, "block": block,
						 "max_score": part.max() if len(part) else np.nan,
						 "n_scores": int(len(part))})
	summary = pd.DataFrame(rows)
	thresholds = summary.groupby("block")["max_score"].quantile(0.99).rename("threshold")
	summary = summary.join(thresholds, on="block")
	summary["alarm"] = np.select(
		[summary["max_score"].isna(), summary["max_score"] >= summary["threshold"]],
		["keine Daten", "Alarm"], default="kein Alarm",
	)

	fig, axes = plt.subplots(len(scores), 1, figsize=(16, 2.8 * len(scores)), sharex=True, squeeze=False)
	for ax, block in zip(axes[:, 0], scores):
		part = summary[summary["block"] == block]
		colors = part["alarm"].map({"Alarm": "firebrick", "kein Alarm": "steelblue", "keine Daten": "lightgray"})
		ax.bar(part["event"], part["max_score"].fillna(0), color=colors)
		ax.axhline(part["threshold"].iloc[0], color="black", ls="--", lw=1)
		ax.set_ylabel(block)
		ax.grid(axis="y", alpha=0.3)
	axes[-1, 0].set_xlabel("SBL-Event")
	fig.suptitle(f"Maximaler Anomaly Score je Block ({lookback} vor SBL)")
	plt.tight_layout()
	plt.show()
	return summary


def plot_feature_heatmaps(feature_tables: dict[str, pd.DataFrame]):
	"""Plot one Seaborn heatmap per block from feature score tables."""
	figures = {}
	for block, table in feature_tables.items():
		heatmap = table.pivot(index="feature", columns="event", values="max_abs_z")
		fig, ax = plt.subplots(figsize=(16, max(4, 0.35 * len(heatmap))))
		sns.heatmap(heatmap, cmap="YlOrRd", mask=heatmap.isna(), linewidths=0.25,
					cbar_kws={"label": "max. |robuster Z-Score|"}, ax=ax)
		ax.set_title(f"{block}: Feature-Abweichungen vor SBL")
		ax.set_xlabel("SBL-Event")
		ax.set_ylabel("Feature")
		plt.tight_layout()
		plt.show()
		figures[block] = fig
	return figures


def plot_feature_scores(feature_tables: dict[str, pd.DataFrame], sbl_times,
						block=None, feature=None, sbl_indices=None,
						before=None, after=None, yscale=None):
	"""Plot robust feature deviations separately, one panel per selected feature."""
	selected = feature_tables if block is None else {block: feature_tables[block]}
	sbl = pd.to_datetime(pd.Series(sbl_times)).dropna().sort_values().reset_index(drop=True)
	selected_sbl = sbl if sbl_indices is None else sbl.iloc[list(sbl_indices)]
	start = pd.Timedelta(before) if before is not None else None
	end = pd.Timedelta(after) if after is not None else None
	panels = []
	for block_name, table in selected.items():
		for feature_name, part in table.groupby("feature"):
			if feature is None or feature_name == feature:
				panels.append((block_name, feature_name, part))
	if not panels:
		raise ValueError("No matching block/feature found.")

	fig, axes = plt.subplots(len(panels), 1, figsize=(16, max(3, 2.6 * len(panels))),
							 sharex=True, squeeze=False)
	for ax, (block_name, feature_name, part) in zip(axes[:, 0], panels):
		plot_part = part[part["event"].isin(list(sbl_indices))] if sbl_indices is not None else part
		# Feature tables contain event-level values, so x is the selected SBL index.
		ax.plot(plot_part["event"], plot_part["max_abs_z"], marker="o", lw=0.9,
				color="steelblue", label="max. |robuster Z-Score|")
		threshold = plot_part["max_abs_z"].quantile(0.99)
		ax.axhline(threshold, color="firebrick", ls="--", lw=1, label="99%-Schwelle")
		ax.set_ylabel(f"{block_name}\n{feature_name}", fontsize=8)
		if yscale is not None:
			ax.set_yscale(yscale)
		ax.grid(alpha=0.3)
		ax.legend(loc="upper right", fontsize=7)
	axes[-1, 0].set_xlabel("SBL-Event")
	fig.suptitle("Feature-Anomaly-Scores pro SBL-Event")
	plt.tight_layout()
	plt.show()
	return fig, axes


def run_pipeline(dataset_dir="Dataset", sbl_path="Dataset/SBL_timestamps_Germany.csv",
				 lookback="10min", buffer="30s") -> dict:
	"""Load, label, train, score, and plot all blocks."""
	blocks = load_all_blocks(dataset_dir)
	sbl = load_sbl_timestamps(sbl_path)["timestamp"]
	labeled = {name: add_sbl_labels(df, sbl, lookback, buffer) for name, df in blocks.items()}
	for name, df in labeled.items():
		df.attrs["sbl_times"] = pd.to_datetime(pd.Series(sbl)).dropna().sort_values().tolist()

	trained = train_block_models(labeled)
	feature_tables = {name: robust_feature_scores(labeled[name], lookback)
					  for name in labeled}
	block_summary = plot_block_scores(trained["scores"], sbl, lookback)
	plot_feature_heatmaps(feature_tables)
	return {"blocks": blocks, "labeled": labeled, "models": trained["models"],
			"scores": trained["scores"], "block_summary": block_summary,
			"feature_tables": feature_tables}


if __name__ == "__main__":
	run_pipeline()
