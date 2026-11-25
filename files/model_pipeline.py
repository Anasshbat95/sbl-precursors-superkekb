"""
model_pipeline.py - Supervised risk/normal classifier (RandomForest) trained on
a single labeled block (output of sbl_labeling.add_sbl_features_and_label),
plus its standard evaluation plots.

This is the simple single-block baseline from the notebook. For combining
several blocks without merging their raw rows, see sbl_anomaly.py instead.

Usage:
    from model_pipeline import build_pipeline, train_and_evaluate

    pipeline = build_pipeline()
    result = train_and_evaluate(pipeline, beam_ds, target_col="target",
                                 drop_cols=("timestamp", "sbl_index"))
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CyclicalTimestampFeatures(BaseEstimator, TransformerMixin):
    """Turn a 'timestamp' column into hour_sin/hour_cos (cyclical encoding), then drop it."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        ts = pd.to_datetime(X["timestamp"])
        hours = ts.dt.hour + ts.dt.minute / 60
        X["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        X["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        return X.drop(columns=["timestamp"])


def build_pipeline(categorical_features=(), n_estimators=300, random_state=42) -> Pipeline:
    """RandomForest pipeline: cyclical time features -> impute/scale/one-hot -> classifier.

    sbl_labeling.add_sbl_features_and_label (the version used in this project) does not add
    a categorical 'sbl_index' column, so the default here has no categorical features. Pass
    categorical_features=("sbl_index",) only if your labeling function adds that column.
    """
    def make_preprocessor(numeric_features):
        return ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
             numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             list(categorical_features)),
        ])

    # numeric_features is filled in by train_and_evaluate once the columns are known
    return Pipeline([
        ("cyclical", CyclicalTimestampFeatures()),
        ("preprocessor", "passthrough"),  # replaced in train_and_evaluate
        ("model", RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators,
                                         random_state=random_state)),
    ])


def train_and_evaluate(pipeline: Pipeline, df: pd.DataFrame, target_col="target",
                       categorical_features=(), drop_cols=("dist_to_sbl_sec", "min_abs_dist",
                                                            "sbl_diff_sec", "sbl_index"),
                       test_size=0.2, random_state=42, plot=True) -> dict:
    """Fit the pipeline, print accuracy, and (optionally) show confusion matrix / ROC / PR / feature-importance plots.

    Returns a dict with the fitted pipeline, the test split and predictions.
    """
    cat = list(categorical_features)
    X = df.drop(columns=[target_col, *drop_cols], errors="ignore")
    y = df[target_col]

    numeric_features = [c for c in X.columns if c not in cat and c != "timestamp"]
    pipeline.set_params(preprocessor=ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
         numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
         cat),
    ]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                         random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = pipeline.score(X_test, y_test)
    print(f"Accuracy: {acc:.3f}")

    if plot:
        _plot_confusion_matrix(y_test, y_pred)
        _plot_roc_and_pr(pipeline, X_test, y_test)
        _plot_feature_importance(pipeline)

    return dict(pipeline=pipeline, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
               y_pred=y_pred, accuracy=acc)


def _plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.title("True vs. predicted")
    plt.show()


def _plot_roc_and_pr(pipeline, X_test, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax1)
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax1.set_title("ROC curve")
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=ax2)
    ax2.set_title("Precision-Recall curve")
    plt.tight_layout()
    plt.show()


def _plot_feature_importance(pipeline, top_n=15):
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    plt.barh([names[i] for i in order][::-1], importances[order][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} features")
    plt.tight_layout()
    plt.show()
