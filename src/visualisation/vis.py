from typing import Any, Mapping, Optional

import numpy as np
from frozendict import frozendict
from geometry import SE2_from_xytheta
from plotly.graph_objs import Figure
from scipy import interpolate

from map.zoo import Track

import plotly.graph_objects as go
from vehicle import vehicles_pool
from vehicle.structures import PlayerName, CarParams
from PIL import Image
from numpy import asarray
from .utils import get_steering_angles


class Visualization:
    """ Visualization for the driving games"""

    fig: Any

    def __init__(self, map: Track, vehicles: Mapping[PlayerName, CarParams] = frozendict(vehicles_pool)):
        self.map = map
        self.gokarts = vehicles
        self._degree = 2
        self.gokart_color = "cornflowerblue"
        self.wheel_color = "lightblue"

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
                name="Track background",
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

    def plot_track(self, fig: Optional[Figure] = None) -> Figure:
        if fig is None:
            fig = go.Figure()
        points = self.map.spline.as_np_array()

        tck, _ = interpolate.splprep(points.T, k=self._degree, per=1)
        u = np.linspace(0, 1, 1000, endpoint=True)
        center_x, center_y, width = interpolate.splev(u, tck)
        centerline = np.column_stack([center_x, center_y])
        dx, dy, _ = interpolate.splev(u, tck, der=1, ext=0)
        normal = np.column_stack([-dy, dx])
        normal_unit = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]

        left_bound = centerline - width[:, np.newaxis] * normal_unit
        right_bound = centerline + width[:, np.newaxis] * normal_unit

        # Constants
        # fig.add_trace(
        #     go.Scatter(
        #         x=centerline[:, 0],
        #         y=centerline[:, 1],
        #         line=dict(color="firebrick", width=1, dash="dot"),
        #         mode="lines",
        #         name="Centerline",
        #     )
        # )
        # fig.add_trace(
        #     go.Scatter(
        #         x=left_bound[:, 0],
        #         y=left_bound[:, 1],
        #         line=dict(color="firebrick", width=4),
        #         mode="lines",
        #         name="Left bound",
        #     )
        # )
        # fig.add_trace(
        #     go.Scatter(
        #         x=right_bound[:, 0],
        #         y=right_bound[:, 1],
        #         line=dict(color="firebrick", width=4),
        #         mode="lines",
        #         name="Right bound",
        #     )
        # )
        return fig

    def plot_vehicle(self, x, y, psi, beta, fig: Figure, gk_name: PlayerName) -> Figure:
        pose = SE2_from_xytheta([x, y, psi])
        xys = self.gokarts[gk_name].get_outline()
        points = np.row_stack([xys, np.ones(xys.shape[1])])
        gk = pose @ points

        wheel_pos = self.gokarts[gk_name].geometry.get_wheel_positions()
        wheel_pos = np.row_stack([wheel_pos, np.ones(wheel_pos.shape[1])])
        wheel_pos_gk = pose @ wheel_pos
        delta_11, delta_12 = get_steering_angles(beta)
        wheel_pose_11 = SE2_from_xytheta([wheel_pos_gk[0, 0], wheel_pos_gk[1, 0], psi + delta_11])
        wheel_pose_12 = SE2_from_xytheta([wheel_pos_gk[0, 1], wheel_pos_gk[1, 1], psi + delta_12])
        wheel_pose_21 = SE2_from_xytheta([wheel_pos_gk[0, 2], wheel_pos_gk[1, 2], psi])
        wheel_pose_22 = SE2_from_xytheta([wheel_pos_gk[0, 3], wheel_pos_gk[1, 3], psi])
        front_wheel = self.gokarts[gk_name].front_tires.get_outline()
        rear_wheel = self.gokarts[gk_name].rear_tires.get_outline()
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
                line=dict(color=self.gokart_color, width=1),
                opacity=0.5,
                fill="toself",
                fillcolor=self.gokart_color,
                mode="lines",
                name="kart",
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

    def plot_prediction(self, x, y, psi, ab, fig: Figure) -> Figure:
        def pred_color(x):
            return "lime" if x > 0 else "red"

        fig.add_trace(
            go.Scatter(
                x=x[:],
                y=y[:],
                line=dict(color=self.gokart_color, width=1, dash="dot"),
                marker=dict(color=list(map(pred_color, ab[:]))),
                mode="lines+markers",
                name="Mpc plan",
            )
        )
        return fig

    def plot_prediction_triangle(self, x, y, psi, ab, fig: Figure) -> Figure:
        triangle = np.array(
            [
                [0, 0.4, 0],
                [-0.2, 0, 0.2],
                [1, 1, 1]
            ]
        )

        prediction_acc = np.array([[None], [None], [None]])
        prediction_dec = np.array([[None], [None], [None]])

        for i in range(len(x)):
            pose = SE2_from_xytheta([x[i], y[i], psi[i]])
            pred = pose @ triangle

            if ab[i] > 0:
                prediction_acc = np.hstack((prediction_acc, pred, np.array([[None], [None], [None]])))
            else:
                prediction_dec = np.hstack((prediction_dec, pred, np.array([[None], [None], [None]])))

        fig.add_trace(
            go.Scatter(
                x=prediction_acc[0, :],
                y=prediction_acc[1, :],
                mode="lines",
                line=dict(color="lime"),
                fill="toself",
                fillcolor="lime",
                name="Mpc plan acc",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=prediction_dec[0, :],
                y=prediction_dec[1, :],
                mode="lines",
                line=dict(color="red"),
                fill="toself",
                fillcolor="red",
                name="Mpc plan dec",
            )
        )

        return fig
