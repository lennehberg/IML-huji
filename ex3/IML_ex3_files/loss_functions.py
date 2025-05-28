import numpy as np


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    errors = np.sum(y_true != y_pred)
    return errors / y_true.shape[0] if normalize else errors


def weighted_misclassiffication_error(y_true: np.ndarray, y_pred: np.ndarray, D: np.ndarray):
    """
    Calculate misclassification loss with weights on samples

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    D: ndarray of shape (n_samples, )
        Distribution weights of samples

    Returns
    -------
    Weighted misclassification of given predictions
    """
    error_indices = np.where(y_true != y_pred)
    return np.sum(D[error_indices])

