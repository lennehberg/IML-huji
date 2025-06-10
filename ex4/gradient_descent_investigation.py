import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type, Any

from sklearn.metrics import roc_curve, auc

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR

from cross_validate import cross_validate
from loss_functions import misclassification_error

# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

import plotly.graph_objects as go
import matplotlib.pyplot as plt


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[GradientDescent, ...], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[GradientDescent, ...], None]
        Callback function to be passed to the GradientDescent class, recording the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs: Any):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"].copy())

    return callback, values, weights

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        lr = FixedLR(eta)
        for module_cls in [L1, L2]:

            # initialize module
            module = module_cls(weights=init.copy())
            module_name = module_cls.__name__
            callback, values, weights = get_gd_state_recorder_callback()

            # run gradient decent
            gd = GradientDescent(lr, tol=1e-5, max_iter=1000, out_type='last', callback=callback)
            gd.fit(module, np.array([]), np.array([])) # X, y are ignored in L1, L2

            # convert weights to ndarray for plot
            descent_path = np.array(weights)
            # Plot the descent path
            fig = plot_descent_path(module_cls, descent_path,
                                    title=f"{module_name} with eta={eta}")
            fig.show()




def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    # Initialize and fit logistic regression model
    logi_regressor = LogisticRegression(alpha=.5)
    logi_regressor.fit(X_train.to_numpy(), y_train.to_numpy())

    # Predict probabilities for test set
    y_scores = logi_regressor.predict_proba(X_test.to_numpy())

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('plots/roc.jpeg')

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter


    # fit model on different lambdas, choose best
    t_scores, v_scores = [], []
    lams = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for lam in lams:
        l1_logi_regressor = LogisticRegression(penalty="l1", lam=lam)
        t_score, v_score = cross_validate(l1_logi_regressor, X_test.to_numpy(), y_test.to_numpy(), scoring=misclassification_error)
        t_scores.append(t_score)
        v_scores.append(v_score)

    # find the lambda with the best validation score
    lam_ind = np.argmax(v_scores)
    print(lams[lam_ind])

    l1_logi_regressor = LogisticRegression(penalty="l1", lam=lams[lam_ind])
    l1_logi_regressor.fit(X_train.to_numpy(), y_train.to_numpy())
    y_scores = l1_logi_regressor.predict_proba(X_test.to_numpy())

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('plots/roc_regu.jpeg')





if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    fit_logistic_regression()
