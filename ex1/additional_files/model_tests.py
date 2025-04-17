import unittest
import plotly.graph_objs as go

import numpy as np

import linear_regression
from ex1.additional_files.linear_regression import LinearRegression


def test_model_no_intercept():
    w0, w1 = 0, 2

    x = np.linspace(0, 100, 10)
    y = w1*x + w0

    fig = go.Figure([go.Scatter(x=x, y=y, name="Real Model", showlegend=True,
                                marker=dict(color="black", opacity=.7), line=dict(color="black", dash="dash", width=1))],
                layout=go.Layout(title=r"$\text{(1) Simulated Data}$",
                                 xaxis={"title": "x - Explanatory Variable"},
                                 yaxis={"title": "y - Response"},
                                 height=400))

    fig.show()

    model = LinearRegression(include_intercept=False)

if __name__ == '__main__':
    test_model_no_intercept()
