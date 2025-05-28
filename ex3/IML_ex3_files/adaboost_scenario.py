import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

from adaboost import AdaBoost
from decision_stump import DecisionStump

def plot_loss_iterations(loss_values: np.ndarray, iterations: np.ndarray, title):
    """
        Plots the loss as a function of iterations.

        Parameters:
        - loss_values (np.ndarray): Array of loss values.
        - iterations (np.ndarray): Array of iteration numbers corresponding to the loss values.
        """
    if len(loss_values) != len(iterations):
        raise ValueError("loss_values and iterations must be the same length.")

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, loss_values, marker='o', linestyle='-', color='blue')
    plt.title(f"{title} Loss vs. Learners")
    plt.xlabel("Learners")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/adaboost/{title}_loss_learners')


def plot_decision_boundaries(ada_model: AdaBoost, X: np.ndarray, y: np.ndarray, T: list, lims: np.ndarray):
    """
    Plot decision boundaries of an AdaBoost model at various iterations.

    Parameters:
    - ada_model (AdaBoost): Trained AdaBoost model.
    - X (np.ndarray): Input data of shape (n_samples, 2).
    - y (np.ndarray): Labels of input data.
    - T (list): List of iteration counts to plot.
    - lims (np.ndarray): Limits for the plot in the form [[x_min, x_max], [y_min, y_max]].
    """
    xx, yy = np.meshgrid(
        np.linspace(lims[0, 0], lims[0, 1], 500),
        np.linspace(lims[1, 0], lims[1, 1], 500)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, len(T), figsize=(5 * len(T), 4))
    if len(T) == 1:
        axes = [axes]

    for i, t in enumerate(T):
        Z = ada_model.partial_predict(grid, t)
        Z = Z.reshape(xx.shape)

        axes[i].contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
        axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolor='k')
        axes[i].set_title(f"AdaBoost Decision Boundary (t = {t})")
        axes[i].set_xlim(lims[0])
        axes[i].set_ylim(lims[1])
        axes[i].set_xlabel("x1")
        axes[i].set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(f'plots/adaboost/des_bounds')



def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def plot_sample_weights_and_decision_surface(ada_model: AdaBoost, X: np.ndarray, y: np.ndarray, lims: np.ndarray):
    """
    Plot decision surface using full AdaBoost model and training samples with sizes proportional to final weights.

    Parameters:
    - ada_model (AdaBoost): Trained AdaBoost model.
    - X (np.ndarray): Training data (n_samples, 2).
    - y (np.ndarray): Training labels.
    - lims (np.ndarray): Plot limits [[x_min, x_max], [y_min, y_max]].
    """

    # Get final sample weights and scale
    D_T = ada_model.D_[-1]  # You might need to adjust this if named differently
    scaled_weights = D_T / np.max(D_T) * 5

    # Define predict function for full model
    def full_predict(X_grid):
        return ada_model.predict(X_grid)

    # Decision surface
    surface = decision_surface(full_predict, xrange=lims[0], yrange=lims[1], dotted=False, showscale=True)

    # Scatter plot of training points
    scatter = go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(
            size=scaled_weights,
            color=y,
            colorscale=custom,
            line=dict(width=0.5, color='black'),
            sizemode='area',
            sizeref=2.*max(scaled_weights)/(40.**2),  # helps size normalization
            sizemin=2
        ),
        showlegend=False,
        hoverinfo='skip'
    )

    # Compose and show figure
    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(title="Final Sample Weights and Decision Surface (Full Ensemble)", width=600, height=500)
    fig.write_image('plots/adaboost/weight_des')



def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    # init adaboost instance
    ada_b = AdaBoost(DecisionStump, n_learners)
    # fit adaboost models to generated data
    ada_b.fit(train_X, train_y)

    # get loss for all partial predictions
    iterations = []
    n_loss_train = []
    n_loss_test = []
    for t in range(n_learners):
        n_loss_train.append(ada_b.partial_loss(train_X, train_y, t))
        n_loss_test.append(ada_b.partial_loss(test_X, test_y, t))
        iterations.append(t)
    # plot loss curve
    plot_loss_iterations(np.array(n_loss_train), np.array(iterations), "train")
    plot_loss_iterations(np.array(n_loss_test), np.array(iterations), "test")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    plot_decision_boundaries(ada_b, test_X, test_y, T, lims)

    # Question 3: Decision surface of best performing ensemble
    # Find best iteration based on test loss
    best_t = np.argmin(n_loss_test)
    print(f"Lowest test loss at iteration {best_t}, loss = {n_loss_test[best_t]:.4f}")

    # Plot decision boundary using best_t
    plot_decision_boundaries(ada_b, test_X, test_y, [best_t], lims)

    # Question 4: Decision surface with weighted samples
    plot_sample_weights_and_decision_surface(ada_b, train_X, train_y, lims)



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)