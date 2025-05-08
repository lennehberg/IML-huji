import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


### HELPER FUNCTIONS ###
def_mean = np.array([0, 0])
def_cov = np.array([[1, 0.5], [0.5, 1]])

def plot_svm_vs_true(X, y, clf, m, C, save_path, ax=None):

    # Create a meshgrid for plotting decision boundaries
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict on the mesh for both the true function and the classifier
    w = np.array([-0.6, 0.4])
    true_pred = np.sign(grid @ w)
    clf_pred = clf.predict(grid)

    # Setup plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the background decision regions
    ax.contourf(xx, yy, true_pred.reshape(xx.shape), alpha=0.2, cmap=ListedColormap(['red', 'blue']))
    ax.contour(xx, yy, clf_pred.reshape(xx.shape), levels=[0], colors='k', linestyles='--')

    # Plot training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'blue']), edgecolors='k')

    ax.set_title(f"SVM vs True Boundary (m={m}, C={C})")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True)

    # Show or save
    if save_path is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# Add here any helper functions which you think will be useful
def f(x):
    w = np.array([-0.6, 0.4])
    ret = np.sign(np.dot(x, w))
    if ret == 0:
        return 1
    else:
        return ret

def assign_label(x):
    return f(x)

def make_label_vector(X):
    return np.array([assign_label(x) for x in X])


def generate_gaussian_data(mean=def_mean, cov=def_cov, size=500):
    X = np.random.multivariate_normal(mean, cov, size)
    y = make_label_vector(X)

    return X, y

### Exercise Solution ###

def pratical_1_runner(save_path=None):
    # generate data and train / test split
    X, y = generate_gaussian_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # go over {5, 10, 20, 100} samples with regularization {0.1, 1, 5, 10, 100}
    sample_sizes = [5, 10, 20, 100]
    reg_values = [0.1, 1, 5, 10, 100]
    for m in sample_sizes:
        for c in reg_values:

            # Sample m samples from training set
            indices = np.random.choice(len(X_train), size=m, replace=False)
            X_sampled = X_train[indices]
            y_sampled = y_train[indices]

            # Fit a model with regularization c
            clf = SVC(kernel='linear', C=c)
            clf.fit(X_sampled, y_sampled)
            # predict on test set
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            # plot results
            print(f"Accuracy for {m} samples and C={c}: {acc}")
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                plot_file = os.path.join(save_path, f"svm_m{m}_C{c}.png")
            else:
                plot_file = None

            plot_svm_vs_true(X_test, y_test, clf, m, c, plot_file)


def practical_2_runner(save_path=None):
    pass


if __name__ == "__main__":
    path = "plots"
    path_1 = f"{path}/practical_1"
    path_2 = f"{path}/practical_2"
    pratical_1_runner(save_path=path_1)
    practical_2_runner(save_path=path_2)