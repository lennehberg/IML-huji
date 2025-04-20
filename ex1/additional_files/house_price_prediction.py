from ex1.additional_files.linear_regression import LinearRegression
from typing import NoReturn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    Preprocess training data.

    Parameters
    ----------
    X : pd.DataFrame
        The loaded data.
    y : pd.Series
        Target values (house prices).

    Returns
    -------
    X, y : pd.DataFrame, pd.Series
        A clean, preprocessed version of the data.
    """
    # Ensure all numeric and clean missing values
    X['floors'] = pd.to_numeric(X['floors'], errors="coerce")
    X = X.dropna()
    y = y.loc[X.index].copy()

    # Remove invalid rows
    invalid_mask = (
        ((X['yr_built'] > X['yr_renovated']) & (X['yr_renovated'] != 0)) |
        (X['bedrooms'] <= 0) |
        (X['sqft_living'] <= 0) |
        (X['sqft_lot'] <= 0) |
        (X['sqft_above'] <= 0) |
        (X['sqft_living15'] <= 0) |
        (X['sqft_lot15'] <= 0) |
        (X['floors'] <= 0) |
        (~X['waterfront'].isin([0, 1])) |
        ((X['view'] < 0) | (X['view'] > 4)) |
        ((X['condition'] < 1) | (X['condition'] > 5)) |
        ((X['grade'] < 1) | (X['grade'] > 13))
    )
    X = X[~invalid_mask]
    y = y[~invalid_mask]

    # Final cleanup of NaNs
    X = X[y.notna()]
    y = y[y.notna()]

    # Engineering features
    X['yr_built'] = X['yr_built'] - X['yr_built'].min()
    X['yr_sold'] = X['date'].str[:4].astype(int)
    X['has_basement'] = (X['sqft_basement'] > 0).astype(int)

    # add multi-floor feature
    # X['is_multi_floor'] = (X['floors'] > 1).astype(int)
    # add luxury feature
    # X['is_luxury'] = ((X['grade'] >= 10) | (X['sqft_living'] > 4000)).astype(int)
    # add sqft_per_room feature
    X['sqft_per_room'] = X['sqft_living'] / (X['bedrooms'] + X['bathrooms'] + 1e-5)
    # location feature
    X['is_premium_location'] = ((X['waterfront'] == 1) | (X['view'] >= 3)).astype(int)

    # X['is_premium_luxury'] = ((X['is_luxury'] >= 1) & (X['is_premium_location'] >= 1)).astype(int)

    X['room_density'] = (X['bedrooms'] + X['bathrooms']) / (X['sqft_living'] + 1e-5)

    X['sqft_grade'] = X['sqft_living'] * X['grade']

    # X['sqft_per_floor'] = X['sqft_living']  / X['floors']

    # Stretch sqft columns to [0, 20] for values > 20
    sqft_cols = [
        'sqft_living', 'sqft_lot',
        'sqft_above', 'sqft_basement',
        'sqft_living15', 'sqft_lot15',
        'sqft_per_room', 'sqft_grade'
    ]
    for col in sqft_cols:
        mask = X[col] > 20
        if mask.any():
            vmin, vmax = X.loc[mask, col].min(), X.loc[mask, col].max()
            if vmax != vmin:
                X.loc[mask, col] = (X.loc[mask, col] - vmin) / (vmax - vmin) * 20

    # Drop irrelevant or redundant columns
    X = X.drop([
        "id", "date", "sqft_lot", "condition", "yr_built",
        "yr_renovated", "yr_sold", "long", "sqft_lot15",
        "has_basement", "floors", "waterfront", "lat", "bedrooms",
        "sqft_basement", "is_premium_location", "view", "bathrooms"
    ], axis=1)

    X = X.fillna(0)
    return X, y



def preprocess_test(X: pd.DataFrame):
    """
    Preprocess test data. You are not allowed to remove rows from X, only edit columns.

    Parameters
    ----------
    X : pd.DataFrame
        The loaded test data.

    Returns
    -------
    X : pd.DataFrame
        A preprocessed version of the test data.
    """
    # Fill missing values and standardize date format
    X = X.fillna(0)
    X['date'] = X['date'].astype(str)
    X['yr_sold'] = X['date'].str[:4].replace('', '0000').astype(int)

    # Add has_basement feature
    X['has_basement'] = (X['sqft_basement'] > 0).astype(int)

    # add multi-floor feature
    # X['is_multi_floor'] = (X['floors'] > 1).astype(int)
    # add luxury feature
    # X['is_luxury'] = ((X['grade'] >= 10) | (X['sqft_living'] > 4000)).astype(int)
    # add sqft_per_room feature
    X['sqft_per_room'] = X['sqft_living'] / (X['bedrooms'] + X['bathrooms'] + 1e-5)
    # location feature
    X['is_premium_location'] = ((X['waterfront'] == 1) | (X['view'] >= 3)).astype(int)

    # X['is_premium_luxury'] = ((X['is_luxury'] >= 1) & (X['is_premium_location'] >= 1)).astype(int)

    X['room_density'] = (X['bedrooms'] + X['bathrooms']) / (X['sqft_living'] + 1e-5)

    X['sqft_grade'] = X['sqft_living'] * X['grade']


    # X['sqft_per_floor'] = X['sqft_living'] / X['floors']

    # Stretch sqft columns to [0, 20] for values > 20
    sqft_cols = [
        'sqft_living', 'sqft_lot',
        'sqft_above', 'sqft_basement',
        'sqft_living15', 'sqft_lot15',
        'sqft_per_room', 'sqft_grade'
    ]
    for col in sqft_cols:
        mask = X[col] > 20
        if mask.any():
            vmin, vmax = X.loc[mask, col].min(), X.loc[mask, col].max()
            if vmax != vmin:
                X.loc[mask, col] = (X.loc[mask, col] - vmin) / (vmax - vmin) * 20

    # Normalize yr_built
    X['yr_built'] = X['yr_built'] - X['yr_built'].min()

    # Drop irrelevant or redundant columns
    X = X.drop([
        "id", "date", "sqft_lot", "condition", "yr_built",
        "yr_renovated", "yr_sold", "long", "sqft_lot15",
        "has_basement", "floors", "waterfront", "lat", "bedrooms",
        "sqft_basement", "is_premium_location", "view", "bathrooms"
    ], axis=1)

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> None:
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
    feature_names = X.columns

    cov_Xy = X.apply(lambda column: y.cov(column))
    std_X = X.std()
    std_y = y.std()

    pearson_corr = cov_Xy  / (std_X * std_y)
    print(pearson_corr)

    # plot correlation graph
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, pearson_corr, color="blue")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.ylabel("Pearson Correlation")
    plt.title("Pearson Correlation Graph")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show() # debug show

    plt.savefig(f"{output_path}/pear_corr_graph.jpeg")



if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    lin_model = LinearRegression()

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)
    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train, "plots")
    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)
    X_mean = X_train.mean()
    X_std = X_train.std()

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages = range(10, 101)
    mean_loss, std_loss = [], []

    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()

    for p in percentages:
        losses = []
        frac = p / 100

        for i in range(10):
            X_sampled = X_train.sample(frac=frac)
            y_sampled = y_train.loc[X_sampled.index]

            lin_model = LinearRegression(include_intercept=True)
            X_sampled_np, y_sampled_np = X_sampled.to_numpy(), y_sampled.to_numpy()
            lin_model.fit(X_sampled_np, y_sampled_np)

            preds = lin_model.predict(X_test_np)
            print(f"Prediction range: {preds.min()} to {preds.max()}")
            print(f"True range: {y_test.min()} to {y_test.max()}")

            loss = lin_model.loss(X_test_np, y_test_np)
            losses.append(loss)


        mean_loss.append(np.mean(losses))
        print("losses: ", mean_loss)
        std_loss.append(np.std(losses))

    plt.figure(figsize=(10, 6))

    # print(mean_loss, std_loss)

    mean_loss = np.array(mean_loss)
    std_loss = np.array(std_loss)



    plt.plot(percentages, mean_loss, label="Mean Loss")
    plt.fill_between(percentages,
                     mean_loss - 2 * std_loss,
                     mean_loss + 2 * std_loss,
                     alpha=0.2, label="±2 std")
    plt.xlabel("Percentage of Training Data")
    plt.ylabel("Test Loss (MSE)")
    plt.title("Model Performance vs. Training Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/model_performance.png")


