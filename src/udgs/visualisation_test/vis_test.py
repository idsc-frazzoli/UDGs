from udgs.visualisation import vis

from udgs.vehicle import vehicles_pool, VEHICLE1
import plotly.graph_objs as go


def test_plot():
    # Create figure
    fig = go.Figure()

    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plotly.com/python/configuration-options/
    plotter = vis.Visualization(map=winti_001, vehicles=vehicles_pool)
    fig = plotter.plot_map(fig)
    fig = plotter.plot_track(fig)
    fig = plotter.plot_vehicle(20, 20, 0.4, fig, VEHICLE1)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.show()
