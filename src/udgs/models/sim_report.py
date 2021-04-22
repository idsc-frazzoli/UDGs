from reprep import Report, MIME_HTML

from udgs.models.forces_def import PG, LexicographicPG, SolutionMethod
from udgs.models.sim import SimData
from udgs.models.sim_plot import get_interactive_scene, get_solver_stats, get_state_plots, get_input_plots
from udgs.visualisation.utils import id2colors


def make_report(sim_data: SimData, solution_method: SolutionMethod) -> Report:
    report = Report("UDGsExperiment")
    n_players = len(sim_data.players)

    vehicles_vis = get_interactive_scene(sim_data.players)
    report.text(nid="Animation", text=vehicles_vis.to_html(), mime=MIME_HTML)

    if solution_method in (PG, LexicographicPG):
        solver_stats = get_solver_stats(sim_data.solver_it, sim_data.solver_time, sim_data.solver_cost)
        report.text(nid=f"SolverStats{solution_method}", text=solver_stats.to_html(), mime=MIME_HTML)
    # else:
    #     solver_stats = get_solver_stats_ibr(sim_data.solver_it, sim_data.solver_time, sim_data.solver_cost,
    #                                         sim_data.convergence_iter)
    #     solver_stats.show()

    for i in range(n_players):
        states = get_state_plots(sim_data.players[i].x)
        report.text(nid=f"StatesPlayer-{id2colors[i]}", text=states.to_html(), mime=MIME_HTML)

        inputs = get_input_plots(sim_data.players[i].u)
        report.text(nid=f"InputsPlayer-{id2colors[i]}", text=inputs.to_html(), mime=MIME_HTML)
    return report
