import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
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


def plot_decision_boundary(X, y, clf, save_path, title="Classifier Decision Boundary", ax=None, resolution=0.02):
    """
    Plots the decision boundary of a classifier along with the data points.

    Parameters:
    - X: np.ndarray, shape (n_samples, 2)
    - y: np.ndarray, shape (n_samples,)
    - clf: Trained classifier with .predict method
    - title: str, plot title
    - ax: optional matplotlib axis
    - resolution: float, mesh resolution
    """
    # Define color maps
    colors = ['red', 'blue']
    light_colors = ['mistyrose', 'lightblue']
    cmap_light = ListedColormap(light_colors)
    cmap_bold = ListedColormap(colors)

    # Meshgrid setup
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
    ax.set_title(title)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
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


def generate_datasets():
    num_samples = 200
    gauss_co_mat = np.array([[0.5, 0.2], [0.2, 0.5]])
    gauss_mean_1 = np.array([-1, -1])
    gauss_mean_2 = np.array([1, 1])
    X_moons, y_moons = make_moons(n_samples=num_samples, noise=0.2)
    X_circles, y_circles = make_circles(n_samples=num_samples, noise=0.1)
    X_gauss_1, y_gauss_1 = generate_gaussian_data(gauss_mean_1, gauss_co_mat, num_samples)
    X_gauss_2, y_gauss_2 = generate_gaussian_data(gauss_mean_2, gauss_co_mat, num_samples)

    return (X_moons, y_moons), (X_circles, y_circles), (X_gauss_1, y_gauss_1), (X_gauss_2, y_gauss_2)

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
    datasets = generate_datasets()
    split_datasets = []
    for dataset in datasets:
        X, y = dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        split_datasets.append((X_train, X_test, y_train, y_test))

    # Classifier setup
    classifiers = {
        "SVM (λ=5)": SVC(C=1/5),
        "Decision Tree (depth=7)": DecisionTreeClassifier(max_depth=7),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
    }

    dataset_names = ["Moons", "Circles", "Gaussian 1", "Gaussian 2"]

    for (X_train, X_test, y_train, y_test), name in zip(split_datasets, dataset_names):
        print(f"\n--- Results for {name} Dataset ---")
        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{clf_name}: Test Accuracy = {acc:.4f}")
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                plot_file = os.path.join(save_path, f"{clf_name}_{name}.png")
            else:
                plot_file = None
            plot_decision_boundary(X_test, y_test, clf, plot_file, title=f"{name} - {clf_name}")



if __name__ == "__main__":
    path = "plots"
    path_1 = f"{path}/practical_1"
    path_2 = f"{path}/practical_2"
    # pratical_1_runner(save_path=path_1)
    practical_2_runner(save_path=path_2)