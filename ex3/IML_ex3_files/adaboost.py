import numpy as np
from typing import Callable, NoReturn
from base_estimator import BaseEstimator
from loss_functions import misclassification_error
from loss_functions import weighted_misclassiffication_error

class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = [], [], []


    @staticmethod
    def _weigh_model(h, X, y, D):
        # find loss according to D
        loss = h.loss(X, y, D)
        # calculate weight of model
        return (1 / 2) * np.log((1 / loss) - 1)


    @staticmethod
    def _weigh_distribution(D, y, y_pred, w):
        # calculate weights for distribution
        D = D * np.exp(-w * y * y_pred)
        # normalize weights
        return D / np.sum(D)


    def _fit_and_weigh(self, X: np.ndarray, y: np.ndarray, D: np.ndarray):
        """
        Fit the weak learner for the base learner

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        D: ndarray of shape (n_samples, )
            Weights for samples
        """
        # Invoke and fit learner
        h = self.wl_()
        h = h.fit(X, y, D)
        y_pred = h.predict(X)

        # calculate weight of model
        w = self._weigh_model(h, X, y, D)

        # calculate weights for distribution
        D = self._weigh_distribution(D, y, y_pred, w)

        return h, w, D


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # set initial distribution to be uniform
        n_samples, n_features = X.shape
        D = np.full(n_samples, 1 / n_samples)

        # boost the weak learner for iterations
        for t in range(self.iterations_):
            # fit weak learner
            h, w, D = self._fit_and_weigh(X, y, D)
            self.models_.append(h)
            self.weights_.append(w)
            self.D_.append(D.copy())

    def _predict_in_range(self, X, T: int):
        if T > self.iterations_:
            return None

        # predict from models up to T
        y_preds = np.array([model.predict(X) for model in self.models_[:T]])

        # weigh the predictions
        weighted_sum = np.dot(self.weights_[:T], y_preds)
        return np.sign(weighted_sum)

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self._predict_in_range(X, self.iterations_)



    def _loss(self, X: np.ndarray, y: np.ndarray, D: np.ndarray= None) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

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
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self._predict_in_range(X, T)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.partial_predict(X, T)
        return misclassification_error(y, y_pred)