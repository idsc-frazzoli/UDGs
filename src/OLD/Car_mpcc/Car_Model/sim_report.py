from reprep import Report

from Car_mpcc.Car_Model.sim import SimData
from Car_mpcc.Car_Model.sim_plot import get_kart_plot, get_solver_stats, get_state_plots, get_input_plots


def make_report(sim_data: SimData):
    # r = Report("vis")

    kart_viz = get_kart_plot(
        sim_data.x, sim_data.x_pred, sim_data.u, sim_data.u_pred, sim_data.next_spline_points, sim_data.track
    )
    kart_viz.show()

    solver_stats = get_solver_stats(sim_data.solver_it, sim_data.solver_time)
    solver_stats.show()

    states = get_state_plots(sim_data.x)
    states.show()

    inputs = get_input_plots(sim_data.u)
    inputs.show()