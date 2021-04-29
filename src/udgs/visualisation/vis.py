from typing import Any, Mapping, Optional

import numpy as np
import plotly
from frozendict import frozendict
from geometry import SE2_from_xytheta
from matplotlib import colors, cm
from matplotlib.colors import rgb2hex
from plotly.graph_objs import Figure

from udgs.map import Lane

import plotly.graph_objects as go
from udgs.vehicle import vehicles_pool, VehicleType
from udgs.vehicle.structures import VehicleParams
from PIL import Image
from numpy import asarray

from udgs.visualisation.utils import id2colors


class Visualization:
    """ Visualization for the driving games"""

    fig: Any

    def __init__(self, map: Lane, vehicles: Mapping[VehicleType, VehicleParams] = frozendict(vehicles_pool)):
        self.map = map
        self.vehicles = vehicles
        self._degree = 2
        self.wheel_color = "lightblue"
        self.colorscale: str = "RdYlGn"
        self.ab_cmap = cm.get_cmap(self.colorscale)
        self.ab_colors_norm = colors.TwoSlopeNorm(0, -5, 3)
        self.id2colors = id2colors

    @staticmethod
    def triangle_outline():
        return np.array(
            [
                [0, 0.4, 0, 0],
                [-0.15, 0, 0.15, -0.15],
                [1, 1, 1, 1]
            ]
        )

    def plot_map(self, fig: Optional[Figure] = None) -> Figure:
        if fig is None:
            fig = go.Figure()
        if self.map.background.shape[-1] == 4:
            data2 = asarray(self.map.background * 255)
            img = Image.fromarray(data2.astype(np.uint8), mode='RGBA')
        else:
            img = Image.fromarray(self.map.background * 255).convert("RGBA")

        # Constants
        img_width = img.width
        img_height = img.height
        scale_factor = self.map.scale_factor

        # Add invisible scatter trace.
        # This trace is added to help the autoresize logic work.
        fig.add_trace(
            go.Scatter(
                x=[0, img_width * scale_factor],
                y=[0, img_height * scale_factor],
                mode="markers",
                marker_opacity=0,
            )
        )

        # Configure axes
        fig.update_xaxes(visible=False, range=[0, img_width * scale_factor])

        fig.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x",
        )

        # Add image
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                name="World",
                xref="x",
                yref="y",
                opacity=1,
                layer="below",
                sizing="stretch",
                source=img,
            )
        )

        # Configure other layout
        fig.update_layout(
            showlegend=False,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        return fig

    def plot_vehicle(self, x, y, theta, delta, fig: Figure, vehicle_type: VehicleType, player_id: int) -> Figure:
        pose = SE2_from_xytheta([x, y, theta])
        xys = self.vehicles[vehicle_type].get_outline()
        points = np.row_stack([xys, np.ones(xys.shape[1])])
        gk = pose @ points

        wheel_pos = self.vehicles[vehicle_type].geometry.get_wheel_positions()
        wheel_pos = np.row_stack([wheel_pos, np.ones(wheel_pos.shape[1])])
        wheel_pos_vehicle = pose @ wheel_pos
        wheel_pose_11 = SE2_from_xytheta([wheel_pos_vehicle[0, 0], wheel_pos_vehicle[1, 0], theta + delta])
        wheel_pose_12 = SE2_from_xytheta([wheel_pos_vehicle[0, 1], wheel_pos_vehicle[1, 1], theta + delta])
        wheel_pose_21 = SE2_from_xytheta([wheel_pos_vehicle[0, 2], wheel_pos_vehicle[1, 2], theta])
        wheel_pose_22 = SE2_from_xytheta([wheel_pos_vehicle[0, 3], wheel_pos_vehicle[1, 3], theta])
        front_wheel = self.vehicles[vehicle_type].front_tires.get_outline()
        rear_wheel = self.vehicles[vehicle_type].rear_tires.get_outline()
        fwheel_points = np.row_stack([front_wheel, np.ones(front_wheel.shape[1])])
        rwheel_points = np.row_stack([rear_wheel, np.ones(rear_wheel.shape[1])])
        wheel_11 = wheel_pose_11 @ fwheel_points
        wheel_12 = wheel_pose_12 @ fwheel_points
        wheel_21 = wheel_pose_21 @ rwheel_points
        wheel_22 = wheel_pose_22 @ rwheel_points

        fig.add_trace(
            go.Scatter(
                x=gk[0, :],
                y=gk[1, :],
                line=dict(color=self.id2colors[player_id], width=1),
                opacity=0.5,
                fill="toself",
                fillcolor=self.id2colors[player_id],
                mode="lines",
                name=f"{self.id2colors[player_id]}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=wheel_11[0, :],
                y=wheel_11[1, :],
                line=dict(color=self.wheel_color, width=1),
                opacity=0.7,
                fill="toself",
                fillcolor=self.wheel_color,
                mode="lines",
                name="Front wheel [11]"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=wheel_12[0, :],
                y=wheel_12[1, :],
                line=dict(color=self.wheel_color, width=1),
                opacity=0.7,
                fill="toself",
                fillcolor=self.wheel_color,
                mode="lines",
                name="Front wheel [12]"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=wheel_21[0, :],
                y=wheel_21[1, :],
                line=dict(color=self.wheel_color, width=1),
                opacity=0.7,
                fill="toself",
                fillcolor=self.wheel_color,
                mode="lines",
                name="Rear wheel [21]"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=wheel_22[0, :],
                y=wheel_22[1, :],
                line=dict(color=self.wheel_color, width=1),
                opacity=0.7,
                fill="toself",
                fillcolor=self.wheel_color,
                mode="lines",
                name="Rear wheel [22]"
            )
        )
        return fig

    def plot_prediction(self, x, y, theta, ab, fig: Figure) -> Figure:
        for i in range(len(x)):
            pose = SE2_from_xytheta([x[i], y[i], theta[i]])
            pred = pose @ self.triangle_outline()
            color = rgb2hex(self.ab_cmap(self.ab_colors_norm(ab[i])))
            fig.add_trace(
                go.Scatter(
                    x=pred[0, :],
                    y=pred[1, :],
                    mode="lines",
                    line=dict(color=color),
                    fill="toself",
                    name="Prediction",
                    showlegend=False
                )
            )
        # add acceleration colorbar
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale=self.colorscale,
                showscale=True,
                cmin=self.ab_colors_norm.vmin,
                cmax=self.ab_colors_norm.vmax,
                colorbar=dict(thickness=15,
                              title='Acc [m/s^2]',
                              tickvals=[self.ab_colors_norm.vmin, self.ab_colors_norm.vmax],
                              ticktext=[str(self.ab_colors_norm.vmin),
                                        str(self.ab_colors_norm.vmax)]),
            ),
            hoverinfo='none'
        )
        )
        return fig
