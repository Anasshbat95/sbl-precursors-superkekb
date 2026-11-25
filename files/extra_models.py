import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM


def fit_scorer(method, X_train, random_state=42):
    """Fit on train-normal rows (already robust-scaled). Returns f(X) -> score, higher = more anomalous."""
    if method == "mcd":
        m = EllipticEnvelope(support_fraction=0.75, random_state=random_state)
    elif method == "ocsvm":
        m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
    elif method == "sgd_ocsvm":
        m = make_pipeline(Nystroem(gamma=1.0 / X_train.shape[1], n_components=min(300, len(X_train) - 1),
                                   random_state=random_state),
                          SGDOneClassSVM(nu=0.01, random_state=random_state))
    elif method == "lof":
        m = LocalOutlierFactor(n_neighbors=35, novelty=True)   # novelty=True: needed to score NEW rows
    elif method == "iforest":
        m = IsolationForest(n_estimators=200, random_state=random_state)
    else:
        raise ValueError(method)
    m.fit(X_train)
    return lambda X: -m.score_samples(X)
