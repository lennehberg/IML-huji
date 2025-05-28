from typing import Tuple
import numpy as np
from base_estimator import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data. Has functions: fit, predict, loss

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # shuffle for randomness
    fold_sizes = np.full(cv, n_samples // cv)
    fold_sizes[:n_samples % cv] += 1  # distribute the remainder

    current = 0
    train_scores = []
    val_scores = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Clone the estimator to ensure independence between folds
        est = estimator

        est.fit(X_train, y_train)
        train_score = est.loss(X_train, y_train)
        val_score = est.loss(X_val, y_val)

        train_scores.append(train_score)
        val_scores.append(val_score)

        current = stop

    return np.mean(train_scores), np.mean(val_scores)
