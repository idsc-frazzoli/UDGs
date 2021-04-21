from typing import Mapping

import plotly.graph_objects as go
import numpy as np
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from udgs.models.forces_def import params
from udgs.vehicle import vehicles_pool, VEHICLE1
from udgs.visualisation.vis import Visualization
from udgs.models.forces_def.indices import var_descriptions
from udgs.models.sim import SimPlayer


def get_car_plot(sim_data: Mapping[int, SimPlayer]) -> Figure:
    """

    :param sim_data:
    :return:
    """

    # some parameters
    n_players = len(sim_data)
    sim_steps = sim_data[0].x.shape[-1]
    N = sim_data[0].x_pred.shape[1]
    # todo sim data needs a better structuring, it should not be map=sim_data[0].lane
    plotter = Visualization(map=sim_data[0].lane, vehicles=vehicles_pool)
    n_inputs = params.n_inputs
    n_states = params.n_states
    x_idx = params.x_idx
    u_idx = params.u_idx

    # create background traces
    fig = plotter.plot_map()
    n_background_traces = len(fig["data"])
    steps = []

    # add traces that change during simulation steps (gokart and predictions)
    for k in range(sim_steps):  # sim_steps:
        for jj in range(n_players):
            upd_s_idx = - n_inputs
            state_k = (
                sim_data[jj].x[x_idx.X + upd_s_idx, k],
                sim_data[jj].x[x_idx.Y + upd_s_idx, k],
                sim_data[jj].x[x_idx.Theta + upd_s_idx, k],
                sim_data[jj].x[x_idx.Delta + upd_s_idx, k])

            fig = plotter.plot_prediction_triangle(
                x=sim_data[jj].x_pred[x_idx.X + upd_s_idx, :, k],
                y=sim_data[jj].x_pred[x_idx.Y + upd_s_idx, :, k],
                psi=sim_data[jj].x_pred[x_idx.Theta + upd_s_idx, :, k],
                ab=sim_data[jj].x_pred[x_idx.Acc + upd_s_idx, :, k],
                fig=fig,
            )
            fig = plotter.plot_vehicle(state_k[0], state_k[1], state_k[2], state_k[3], fig, VEHICLE1)
            fig = plotter.plot_vehicle(50, 37, 0, 0, fig, VEHICLE1)

    n_step_traces = int((len(fig["data"]) - n_background_traces) / sim_steps)
    for k in range(sim_steps):
        vis_bool = [True] * n_background_traces + [False] * (len(fig.data) - n_background_traces)
        slider_step = dict(label=str(k), method="restyle", args=[{"visible": vis_bool}])
        # Toggle k'th traces to "visible"
        active_trac_idx = int(n_background_traces + k * n_step_traces)
        slider_step["args"][0]["visible"][active_trac_idx: active_trac_idx + n_step_traces] = \
            [True] * n_step_traces
        steps.append(slider_step)

    # do slider logic
    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Simulation step: ", "font": {"size": 18}, "xanchor": "right"},
            len=0.9,
            x=0.1,
            y=0,
            steps=steps,
        )
    ]
    # play and pause buttons
    updatemenus = [
        dict(
            buttons=[
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "quadratic-in-out"},
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}},
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            direction="left",
            pad={"r": 40, "t": 45},
            showactive=False,
            type="buttons",
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top",
        )
    ]

    # update some layout (not working yet)
    fig.update_layout(
        dict(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            title="Simulation",
            hovermode="closest",
            updatemenus=updatemenus,
            sliders=sliders,
        )
    )
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    return fig


def get_solver_stats(solver_it, solver_time, solver_cost) -> Figure:
    n_sim_steps = solver_time.shape[0]
    n_lexi_calls = solver_time.shape[1]
    sim_steps = np.arange(0, n_sim_steps)

    fig = make_subplots(
        rows=3 * n_lexi_calls, cols=2, column_widths=[0.7, 0.3],
        subplot_titles=("# Iterations", "", "Solving time", "", "Cost", "")
    )
    for jj in range(n_lexi_calls):
        # stats about solver iterations
        it_color = "firebrick"
        fig.add_trace(
            go.Scatter(
                x=sim_steps,
                y=solver_it[:, jj],
                line=dict(color=it_color, width=1, dash="dot"),
                mode="lines+markers",
                name="Solver iterations",
            ),
            row=1 + jj * 3,
            col=1,
        )
        fig.add_trace(go.Histogram(y=solver_it[:, jj], marker_color=it_color, opacity=0.75), row=1 + jj * 3, col=2)

        # stats about solving time
        time_color = "blueviolet"
        fig.add_trace(
            go.Scatter(
                x=sim_steps,
                y=solver_time[:, jj],
                line=dict(color=time_color, width=1, dash="dot"),
                mode="lines+markers",
                name="Solver time",
            ),
            row=2 + jj * 3,
            col=1,
        )
        fig.add_trace(go.Histogram(y=solver_time[:, jj], marker_color=time_color, opacity=0.75), row=2 + jj * 3, col=2)
        print(f"Average solving time: {np.average(solver_time[:, jj]):.4f}")

        # stats about solving time
        time_color = "cyan"
        fig.add_trace(
            go.Scatter(
                x=sim_steps,
                y=solver_cost[:, jj],
                line=dict(color=time_color, width=1, dash="dot"),
                mode="lines+markers",
                name="Solver costs",
            ),
            row=3 + jj * 3,
            col=1,
        )
        fig.add_trace(go.Histogram(y=solver_cost[:, jj], marker_color=time_color, opacity=0.75), row=3 + jj * 3, col=2)
    # general layout
    fig.update_layout(title_text="Solver stats")
    return fig


def get_solver_stats_ibr(solver_it, solver_time, solver_cost, convergence_iter) -> Figure:
    n_sim_steps = solver_time.shape[0]
    n_lexi_calls = solver_time.shape[1]
    n_players = solver_time.shape[2]
    n_iter = solver_time.shape[3]
    sim_steps = np.arange(0, n_sim_steps)

    fig = make_subplots(
        rows=3 * n_lexi_calls, cols=2, column_widths=[0.7, 0.3],
        subplot_titles=("# Iterations", "", "Solving time", "", "Cost", "")
    )
    for jj in range(n_lexi_calls):
        # stats about solver iterations
        it_color = "firebrick"
        fig.add_trace(
            go.Scatter(
                x=sim_steps,
                y=solver_it[:, jj],
                line=dict(color=it_color, width=1, dash="dot"),
                mode="lines+markers",
                name="Solver iterations",
            ),
            row=1 + jj * 3,
            col=1,
        )
        fig.add_trace(go.Histogram(y=solver_it[:, jj], marker_color=it_color, opacity=0.75), row=1 + jj * 3, col=2)

        # stats about solving time
        time_color = "blueviolet"
        fig.add_trace(
            go.Scatter(
                x=sim_steps,
                y=solver_time[:, jj],
                line=dict(color=time_color, width=1, dash="dot"),
                mode="lines+markers",
                name="Solver time",
            ),
            row=2 + jj * 3,
            col=1,
        )
        fig.add_trace(go.Histogram(y=solver_time[:, jj], marker_color=time_color, opacity=0.75), row=2 + jj * 3, col=2)
        print(f"Average solving time: {np.average(solver_time[:, jj]):.4f}")

        # stats about solving time
        time_color = "cyan"
        fig.add_trace(
            go.Scatter(
                x=sim_steps,
                y=solver_cost[:, jj],
                line=dict(color=time_color, width=1, dash="dot"),
                mode="lines+markers",
                name="Solver costs",
            ),
            row=3 + jj * 3,
            col=1,
        )
        fig.add_trace(go.Histogram(y=solver_cost[:, jj], marker_color=time_color, opacity=0.75), row=3 + jj * 3, col=2)
    # general layout
    fig.update_layout(title_text="Solver stats")
    return fig


def get_state_plots(states) -> Figure:
    n_sim_steps = states.shape[1]
    sim_steps = np.arange(0, n_sim_steps)
    n_states = params.n_states
    names = []
    units = []

    for state_var in params.x_idx:
        names.append(var_descriptions[state_var].title)
        units.append(var_descriptions[state_var].units)

    fig = make_subplots(
        rows=int(np.ceil(n_states / 2)), cols=2, column_widths=[0.5, 0.5], subplot_titles=names
    )

    for i in range(n_states):
        row = int(np.floor(i / 2)) + 1
        col = np.mod(i, 2) + 1
        fig.add_trace(
            go.Scatter(
                x=sim_steps,
                y=states[i, :],
                line=dict(width=1, dash="dot"),
                mode="lines+markers",
                name=names[i],
            ),
            row=row,
            col=col
        )
        fig.update_yaxes(title_text=units[i], row=row, col=col)

    fig.update_layout(title_text="States")

    return fig


def get_input_plots(inputs) -> Figure:
    n_sim_steps = inputs.shape[1]
    sim_steps = np.arange(0, n_sim_steps)
    n_inputs = params.n_inputs
    names = []
    units = []

    for input_var in params.u_idx:
        names.append(var_descriptions[input_var].title)
        units.append(var_descriptions[input_var].units)

    fig = make_subplots(
        rows=int(np.ceil(n_inputs / 2)), cols=2, column_widths=[0.5, 0.5], subplot_titles=names
    )

    for i in range(n_inputs):
        row = int(np.floor(i / 2)) + 1
        col = np.mod(i, 2) + 1
        fig.add_trace(
            go.Scatter(
                x=sim_steps,
                y=inputs[i, :],
                line=dict(width=1, dash="dot"),
                mode="lines+markers",
                name=names[i],
            ),
            row=row,
            col=col
        )
        fig.update_yaxes(title_text=units[i], row=row, col=col)

    fig.update_layout(title_text="Inputs")

    return fig
