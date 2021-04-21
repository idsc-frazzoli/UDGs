from matplotlib import cm, colors
import numpy as np
from geometry import SE2_from_xytheta
from matplotlib.colors import rgb2hex
from plotly.graph_objs import Figure
import plotly.graph_objects as go


def test_plot():
    n_points = 40
    x = np.linspace(0, 10, n_points)
    y = np.linspace(0, 10, n_points)
    theta = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    ab = np.sin(list(range(n_points)))*5
    triangle = np.array(
            [
                [0, 0.4, 0, 0],
                [-0.15, 0, 0.15, -0.15],
                [1, 1, 1, 1]
            ]
        )
      # min max acceleration

    fig = Figure()

    for i in range(len(x)):
        pose = SE2_from_xytheta([x[i], y[i], theta[i]])
        pred = pose @ triangle


        fig.add_trace(
            go.Scatter(
                x=pred[0, :],
                y=pred[1, :],
                mode="lines",
                line=dict(color="lime", ),
                fill="toself",
                showlegend=False
            )
        )
    fig.show()
