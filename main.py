import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from typing import NamedTuple


DEFAULT_PREDICTION_YEAR = 2024


class PredictionResult(NamedTuple):
    turnover_rate: float
    births: int
    deaths: int
    population: int
    discharges_and_deaths_in_hospital: int
    hospital_beds: int
    total_nurses: int
    leaving_nurses: int

    def __repr__(self):
        return (
            f"Births: {self.births}\n"
            f"Deaths: {self.deaths}\n"
            f"Population: {self.population}\n"
            f"Discharges and Deaths in Hospital: {self.discharges_and_deaths_in_hospital}\n"
            f"Hospital Beds: {self.hospital_beds}\n"
            f"Total Nurses: {self.total_nurses}\n"
            f"Turnover Rate: {round(self.turnover_rate * 100, 2)}%\n"
            f"Leaving Nurses: {self.leaving_nurses} (Total Nurses * Turnover Rate)\n"
        )


def predict(model, original_years, prediction_year):
    prediction = model.predict(
        np.append(original_years, prediction_year).reshape(-1, 1)
    )

    return list(
        map(
            lambda x: PredictionResult(
                turnover_rate=x[0],
                births=int(x[1]),
                deaths=int(x[2]),
                population=int(x[3]),
                discharges_and_deaths_in_hospital=int(x[4]),
                hospital_beds=int(x[5]),
                total_nurses=int(x[6]),
                leaving_nurses=int(x[0] * x[6]),
            ),
            prediction,
        )
    )


def plot_graph(
    title,
    original_years,
    original_leaving_nurses,
    prediction_year,
    prediction_results,
    get_prediction_field=lambda x: x.leaving_nurses,
):
    plt.figure()
    plt.scatter(original_years, original_leaving_nurses)
    plt.scatter(
        [prediction_year], list(map(get_prediction_field, prediction_results))[-1]
    )
    plt.plot(
        np.append(original_years, prediction_year).reshape(-1, 1),
        list(map(get_prediction_field, prediction_results)),
        color="blue",
        linewidth=3,
    )
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Leaving Nurses")
    plt.plot()


# Linear regression
def linear_regression(
    X, Y, original_years, total_nurses, turnover_rate, prediction_year
):
    model = LinearRegression(fit_intercept=True)
    model.fit(X, Y)
    prediction = predict(model, original_years, prediction_year)
    plot_graph(
        "Linear Regression",
        original_years,
        total_nurses * turnover_rate,
        prediction_year,
        prediction,
    )
    print(
        f"Linear Regression Prediction Result for {prediction_year}:",
        prediction[-1],
        sep="\n",
    )


# Ridge regression
def ridge_regression(
    X, Y, original_years, total_nurses, turnover_rate, prediction_year
):
    model = Ridge()
    model.fit(X, Y)
    prediction = predict(model, original_years, prediction_year)
    plot_graph(
        "Ridge Regression",
        original_years,
        total_nurses * turnover_rate,
        prediction_year,
        prediction,
    )
    print(
        f"Ridge Regression Prediction Result for {prediction_year}:",
        prediction[-1],
        sep="\n",
    )


def main(argv):
    prediction_year = DEFAULT_PREDICTION_YEAR if len(argv) == 0 else int(argv[0])
    dataframe = pd.read_csv("dataset.csv")

    # Factors
    year = np.array(dataframe["year"])

    # Prediction result
    turnover_rate = np.array(dataframe["turnover_rate"])
    births = np.array(dataframe["births"])
    deaths = np.array(dataframe["deaths"])
    population = np.array(dataframe["population"])
    discharges_and_deaths_in_hospital = np.array(
        dataframe["discharges_and_deaths_in_hospital"]
    )
    hospital_beds = np.array(dataframe["hospital_beds"])
    total_nurses = np.array(dataframe["total_nurses"])

    # Construct variables X and Y
    X = year.reshape(-1, 1)
    Y = np.concatenate(
        [
            turnover_rate.reshape(-1, 1),
            births.reshape(-1, 1),
            deaths.reshape(-1, 1),
            population.reshape(-1, 1),
            discharges_and_deaths_in_hospital.reshape(-1, 1),
            hospital_beds.reshape(-1, 1),
            total_nurses.reshape(-1, 1),
        ],
        axis=1,
    )

    linear_regression(X, Y, year, total_nurses, turnover_rate, prediction_year)
    ridge_regression(X, Y, year, total_nurses, turnover_rate, prediction_year)

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
