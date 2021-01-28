from tracks.zoo import winti_001
from visualisation import vis

from vehicle import gokart_pool, KITT
import plotly.graph_objs as go


def test_plot():
    # Create figure
    fig = go.Figure()

    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plotly.com/python/configuration-options/
    plotter = vis.Visualization(track=winti_001, gokarts=gokart_pool)
    fig = plotter.plot_map(fig)
    fig = plotter.plot_track(fig)
    fig = plotter.plot_gokart(20, 20, 0.4, fig, KITT)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.show()
