from reprep import Report

from udgs.models.forces_def import PG, LexicographicPG, SolutionMethod
from udgs.models.sim import SimData
from udgs.models.sim_plot import get_interactive_scene, get_solver_stats, get_state_plots, get_input_plots, \
    get_open_loop_animation


def make_report(sim_data: SimData) -> Report:
    report = Report(nid="udgs", caption="Urban Driving Game Experiment report")
    report.add_child(get_interactive_scene(sim_data.players))
    report.add_child(get_open_loop_animation(sim_data.players, 2))

    if sim_data.solution_method in (PG, LexicographicPG):
        report.add_child(get_solver_stats(sim_data))
    # else:
    #     solver_stats = get_solver_stats_ibr(sim_data.solver_it, sim_data.solver_time, sim_data.solver_cost,
    #                                         sim_data.convergence_iter)
    #     solver_stats.show()

    for i in range(len(sim_data.players)):
        report.add_child(get_state_plots(sim_data.players[i].x, i))
        report.add_child(get_input_plots(sim_data.players[i].u, i))
    return report
