from reprep import Report

from Car_mpcc.Car_optimizer.sim import SimData
from Car_mpcc.Car_optimizer.sim_plot import get_car_plot, get_solver_stats, get_state_plots, get_input_plots


def make_report(sim_data: SimData, num_cars):
    # r = Report("vis")

    cars_viz = get_car_plot(
        sim_data.x, sim_data.x_pred, sim_data.u, sim_data.u_pred, sim_data.next_spline_points, num_cars, sim_data.track
    )
    cars_viz.show()

    solver_stats = get_solver_stats(sim_data.solver_it, sim_data.solver_time)
    solver_stats.show()

    states = get_state_plots(sim_data.x, num_cars)
    states.show()

    inputs = get_input_plots(sim_data.u, num_cars)
    inputs.show()
