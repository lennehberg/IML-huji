import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.dropna()
    df = df[df['Temp'] > -70]

    # add day of year column for polynomial features
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month  # Add Month column for easy grouping
    return df


def investigate_israel(X: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data for entries where the country is Israel and plot temperature vs. day of the year.

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix containing temperature data and features.

    y : pd.Series
        Response vector containing temperature values.

    Returns
    -------
    None
    """
    X_filtered = X[X['Country'] == 'Israel']

    # Add Year as a column for color coding
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_filtered['DayOfYear'], X_filtered['Temp'], c=X_filtered['Year'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Year')
    plt.title("Temperature vs. Day of Year for Israel (color-coded by Year)")
    plt.xlabel("Day of Year")
    plt.ylabel("Temperature (°C)")
    plt.grid()
    plt.savefig('plots/temp_israel_dayofyear.jpeg')

    plot_monthly_temperature_std(X, 'Israel')
    
    return X_filtered


def plot_monthly_temperature_std(X: pd.DataFrame, country: str):
    """
    Group data by month and plot the standard deviation of daily temperatures.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing temperature data.

    country : str
        Country for which the plot is generated.

    Returns
    -------
    None
    """
    X_filtered = X[X['Country'] == country]

    # Extract Month from the Date if not present
    if 'Month' not in X_filtered.columns:
        X_filtered['Month'] = X_filtered['Date'].dt.month

    # Group by Month and calculate the standard deviation of temperatures
    monthly_std = X_filtered.groupby('Month')['Temp'].std()

    # Plot the standard deviation
    plt.figure(figsize=(8, 5))
    plt.bar(monthly_std.index, monthly_std.values, color='skyblue', edgecolor='black')
    plt.xticks(np.arange(1, 13), [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ])
    plt.ylabel('Temperature Std Dev (°C)')
    plt.xlabel('Month')
    plt.title(f'Standard Deviation of Daily Temperatures by Month in {country}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'plots/temp_std_{country.lower()}_month.jpeg')


def plot_avg_monthly_temp_with_error_bars(X: pd.DataFrame):
    """
    Group data by country and month, calculate average and standard deviation 
    of temperature, and plot a line plot with error bars.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing temperature data.

    Returns
    -------
    None
    """
    # Group by Country and Month, calculate mean and std of temperature
    grouped = X.groupby(["Country", "Month"])["Temp"].agg(["mean", "std"]).reset_index()

    # Loop through each country and plot its monthly temperature with error bars
    countries = grouped["Country"].unique()
    plt.figure(figsize=(12, 6))

    for country in countries:
        country_data = grouped[grouped["Country"] == country]
        plt.errorbar(
            country_data["Month"],
            country_data["mean"],
            yerr=country_data["std"],
            label=country,
            capsize=5,
            marker="o",
            linestyle="--"
        )

    plt.xticks(np.arange(1, 13), [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ])
    plt.title("Average Monthly Temperature with Error Bars")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.legend(title="Country")
    plt.grid(alpha=0.5)
    plt.savefig('plots/avg_monthly_temp_with_error_bars.jpeg')


def test_model_israel(X_israel: pd.DataFrame):
    """
    Randomly split Israel's data into train and test sets, fit polynomial models
    of varying degrees, and evaluate their performance.

    Parameters
    ----------
    X_israel: pd.DataFrame
        DataFrame containing Israel's temperature data.

    Returns
    -------
    None
    """
    # Split data
    X_israel = X_israel.sample(frac=1, random_state=123)  # Shuffle the data
    train_size = int(0.75 * len(X_israel))
    train_X = X_israel.iloc[:train_size]["DayOfYear"].values
    train_y = X_israel.iloc[:train_size]["Temp"].values
    test_X = X_israel.iloc[train_size:]["DayOfYear"].values
    test_y = X_israel.iloc[train_size:]["Temp"].values

    # Loop through polynomial degrees and compute test loss
    losses = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X, train_y)
        loss = model.loss(test_X, test_y)
        losses.append((k, round(loss, 2)))

    # Print results
    for k, loss in losses:
        print(f"Degree {k}: Test Loss = {loss}")

    # Bar plot of test errors for each polynomial degree
    degrees = [k for k, _ in losses]
    errors = [loss for _, loss in losses]
    plt.figure(figsize=(10, 6))
    plt.bar(degrees, errors, color='lightcoral', edgecolor='black')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Test Error (MSE)")
    plt.title("Test Error by Polynomial Degree")
    plt.xticks(degrees)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig('plots/test_errors_degrees.jpeg')


def train_israel_test_rest(X: pd.DataFrame, best_degree: int = 5):
    """
    Fit the polynomial model over records from Israel and evaluate it on other countries.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with temperature data.

    Returns
    -------
    None
    """
    # Define the countries to evaluate the model on
    other_countries = ["The Netherlands", "South Africa", "Jordan"]

    # Filter Israel dataset and train the model using the chosen degree (e.g., k = 5)
    X_israel = X[X["Country"] == "Israel"]
    train_X = X_israel["DayOfYear"].values
    train_y = X_israel["Temp"].values
    chosen_degree = best_degree  # Chosen "k" value from previous step

    model = PolynomialFitting(chosen_degree)
    model.fit(train_X, train_y)

    # Compute the model loss on each of the other countries
    errors = {}
    for country in other_countries:
        X_country = X[X["Country"] == country]
        test_X = X_country["DayOfYear"].values
        test_y = X_country["Temp"].values
        errors[country] = np.round(model.loss(test_X, test_y), 2)

    # Plot the errors as a bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(errors.keys(), errors.values(), color="lightsalmon", edgecolor="black")
    plt.ylabel("Test Error (MSE)")
    plt.xlabel("Country")
    plt.title(f"Model Errors on Other Countries (Degree {chosen_degree})")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("plots/israel_model_errors_countries.jpeg")


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    X = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    X_israel = investigate_israel(X)

    # Plot monthly temperature standard deviation for Israel
    plot_monthly_temperature_std(X, "Israel")

    # Question 4 - Exploring differences between countries
    plot_avg_monthly_temp_with_error_bars(X)

    # Question 5 - Fitting model for different values of `k`
    test_model_israel(X_israel)

    # Question 6 - Evaluating fitted model on different countries
    train_israel_test_rest(X)
