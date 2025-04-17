from ex1.additional_files.linear_regression import LinearRegression
from typing import NoReturn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    # make sure all values are numeric
    X['floors'] = pd.to_numeric(X['floors'], errors="coerce")
    X = X.dropna() # drop all NaN values from both X and y
    y = y.loc[X.index].copy()
    # make sure all values are withing expected ranges
    invalid_mask = (
        X['yr_built'] < X['yr_renovated'] |
        X['bedrooms'] <= 0 |
        X['price'] <= 0 |
        X['sqft_living'] <= 0 |
        X['sqft_lot'] <= 0 |
        X['sqft_above'] <= 0 |
        X['sqft_living15'] <= 0 |
        X['sqft_lot15'] <= 0 |
        X['floors'] <= 0 |
        (~X['waterfront'].isin([0, 1])) |
        ((X['view'] < 0) | (X['view'] > 4)) |
        ((X['condition'] < 1) | (X['condition'] > 5)) |
        ((X['grade'] < 1) | (X['grade'] > 13))
    )
    # save the indexes that are valid
    X = X[~invalid_mask]
    y = y[~invalid_mask]

    # add a price per square feet parameter
    # X['d_p_sqft'] = y / X['sqft_living']
    # add year sold and renovation difference parameters
    X['yr_sold'] = X['date'].str[:4].astype(int)
    X['renovated_diff'] = np.where(X['yr_renovated'] > 0, X['yr_sold'] - X['yr_renovated'], 0)
    # add more if needed... (hasBasement, )

    # remove irrelevant information
    X = X.drop(["id", "date"], axis=1)
    return X, y


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    # add a price per square feet parameter
    # X['d_p_sqft'] = y / X['sqft_living']
    # add year sold and renovation difference parameters
    X['yr_sold'] = X['date'].str[:4].astype(int)
    X['renovated_diff'] = np.where(X['yr_renovated'] > 0, X['yr_sold'] - X['yr_renovated'], 0)
    # add more if needed... (hasBasement, )

    X = X.drop(["id", "date"], axis=1)
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # save feature names for plot
    feature_names = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                     "floors", "waterfront", "view", "condition", "grade",
                     "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
                     "lat", "long", "sqf_living15", "sqft_lot15",
                     # "d_p_sqft",
                     "yr_sold", "renovated_diff"]

    cov_Xy = X.apply(lambda column: y.cov(column))
    std_X = X.std()
    std_y = y.std()

    pearson_corr = cov_Xy  / std_X * std_y
    print(pearson_corr)

    # plot correlation graph
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, pearson_corr, color="bluesky")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.ylabel("Pearson Correlation")
    plt.title("Pearson Correlation Graph")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show() # debug show

    plt.imsave(f"{output_path}/pear_corr_graph.jpeg")



if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)
    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X, y, "plots")
    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)
    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

