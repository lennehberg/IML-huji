from __future__ import annotations
from typing import Tuple, NoReturn
from base_estimator import BaseEstimator
import numpy as np
from itertools import product
from loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_features = X.shape[1]
        min_error = float('inf')

        # find the best feature which brings the misclassification error to a minimum
        for j in range(n_features):
            feature_values = X[:, j]
            # on this feature, find the best threshhold with the best sign,
            # save if misclasification error is smaller than min_error
            for sign in [-1, 1]:
                threshhold, error = self._find_threshold(feature_values, y, sign)
                if error < min_error:
                    min_error = error
                    self.threshold_ = threshhold
                    self.j_ = j
                    self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        feature_values = X[:, self.j_]
        return np.where(feature_values >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # sort feature values for easier computation
        sorted_indices = np.argsort(values)
        sorted_values, sorted_labes = values[sorted_indices], labels[sorted_indices]

        # find all thresh-holds between 2 consecutive values
        threshholds = np.concatenate([-np.inf],
                                     (sorted_values[1:] + sorted_values[:-1]) / 2,
                                     [np.inf])

        best_threshhold = None
        min_error=float('inf')

        # iterate over all threshholds to find the one that brings the misclassification error to a minimum
        for t in threshholds:
            preds = np.where(values>=t, sign, -sign)
            error = misclassification_error(labels, preds, normalize=True)
            # if the error is smaller then the min error, set the new threshhold, and keep looking for a better minimizer
            if error < min_error:
                best_threshhold = t
                min_error = error

        return best_threshhold, min_error


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred, True)
