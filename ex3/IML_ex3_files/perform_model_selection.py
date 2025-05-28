import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import datasets
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression
from utils import split_train_test


def plot_score_lam(score_values: np.ndarray, lams: np.ndarray, title):
    """
        Plots the loss as a function of iterations.

        Parameters:
        - loss_values (np.ndarray): Array of loss values.
        - iterations (np.ndarray): Array of iteration numbers corresponding to the loss values.
        """
    if len(score_values) != len(lams):
        raise ValueError("loss_values and iterations must be the same length.")

    plt.figure(figsize=(8, 5))
    plt.plot(lams, score_values, marker='o', linestyle='-', color='blue')
    plt.title(f"{title} Scores vs. Lambdas")
    plt.xlabel("Lambda")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    diabetes = datasets.load_diabetes(as_frame=True)
    train_X, train_y, test_X, test_y = split_train_test(diabetes.data, diabetes.target,
                                                            n_samples / diabetes.target.shape[0]) # diabetes has 442 samples according to docs

    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    lams = np.array([_ for _ in range(n_evaluations)])
    lams = 0.2 * lams + 0.0001 # stretch to range 1 / 5 of n_evaluations
    ls_train_scores, ls_val_scores = [], []
    rd_train_scores, rd_val_scores = [], []

    for lam in lams:
        ls = Lasso(lam, True)
        rd = RidgeRegression(lam, True)

        # cross validate each model with current lambda
        ls_train_score, ls_val_score = cross_validate(ls, test_X.to_numpy(),
                                                      test_y.to_numpy(), 5)
        rd_train_score, rd_val_score = cross_validate(rd, test_X.to_numpy(),
                                                      test_y.to_numpy(), 5)
        ls_train_scores.append(ls_train_score)
        ls_val_scores.append(ls_val_score)
        rd_train_scores.append(rd_train_score)
        rd_val_scores.append(rd_val_score)

    plot_score_lam(np.array(ls_train_scores), lams, "Lasso train")
    plot_score_lam(np.array(ls_val_scores), lams, "Lasso evaluation")
    plot_score_lam(np.array(rd_train_scores), lams, "Ridge train")
    plot_score_lam(np.array(rd_val_scores), lams, "Ridge evaluation")


    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ls_lam = float(lams[np.argmin(ls_val_scores)])
    best_rd_lam = float(lams[np.argmin(rd_val_scores)])

    # init new learners and fit them with best lambdas
    lr, ls, rd = LinearRegression(), Lasso(best_ls_lam), RidgeRegression(best_rd_lam)
    lr.fit(train_X.to_numpy(), train_y.to_numpy())
    ls.fit(train_X.to_numpy(), train_y.to_numpy())
    rd.fit(train_X.to_numpy(), train_y.to_numpy())

    print(f"Linear loss: {lr.loss(test_X.to_numpy(), test_y.to_numpy())}"
          f" Lasso loss: {ls.loss(test_X.to_numpy(), test_y.to_numpy())}, lambda: f{best_ls_lam}"
          f" Ridge loss: {rd.loss(test_X.to_numpy(), test_y.to_numpy())}, lambda: f{best_rd_lam}")



if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
