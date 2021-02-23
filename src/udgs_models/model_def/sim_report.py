from reprep import Report

from udgs_models.model_def.sim import SimData
from udgs_models.model_def.sim_plot import get_car_plot, get_solver_stats, get_state_plots, get_input_plots, \
    get_solver_stats_ibr


def make_report(sim_data: SimData, condition):
    # r = Report("vis")
    n_players = len(sim_data.players)

    cars_viz = get_car_plot(sim_data.players)
    cars_viz.show()

    if condition == 0 or condition == 1:
        solver_stats = get_solver_stats(sim_data.solver_it, sim_data.solver_time, sim_data.solver_cost)
        solver_stats.show()
    # else:
    #     solver_stats = get_solver_stats_ibr(sim_data.solver_it, sim_data.solver_time, sim_data.solver_cost,
    #                                         sim_data.convergence_iter)
    #     solver_stats.show()

    for i in range(n_players):
        states = get_state_plots(sim_data.players[i].x)
        states.show()
    
        inputs = get_input_plots(sim_data.players[i].u)
        inputs.show()
