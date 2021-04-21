import argparse
import os.path

from udgs.models.sim import sim_car_model
from udgs.models.forces_def.generate_model import AVAILABLE_METHODS, generate_forces_models, SolutionMethod
from udgs.models.sim_report import make_report


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--generate_solver",
        default=False,
        help="If set to false does not regenerate the solver but it looks for an existing one",
        type=bool,
    )
    p.add_argument(
        "--to_deploy",
        default=False,
        help="If True generates the solver with additional flags for targeting Docker Linux platform",
        type=bool,
    )
    p.add_argument(
        "--n_players",
        default=3,
        help="Number of active players in the scenario. Integer between 1 and 4",
        type=int,
    )
    p.add_argument(
        "--output_dir",
        default="out-docker",
        help="Output folder where the experiment report is placed",
        type=str,
    )
    p.add_argument(
        "--solution_method",
        default="PG",
        help="PG: potential game solution,"
             "LexicographicPG: Lexicographic potential game"
             "IBR: Iterated best response"
             "LexicographicIBR: Lexicographic Iterated best response",
        type=str,
    )
    return p.parse_args()


def _assert_args(args):
    if args.solution_method not in AVAILABLE_METHODS:
        raise ValueError(f"{args.solution_method} is not available. Retry with one amongst {AVAILABLE_METHODS}")
    if args.n_players not in range(1, 5):
        raise ValueError(f"{args.n_players} is not available. Retry with a value between 1 and 4")
    return


def main(generate_solver: bool,
         to_deploy: bool,
         n_players: int,
         solution_method: SolutionMethod,
         output_dir: str):
    # generate forces models definition and solvers
    forces_models = generate_forces_models(generate_solver, to_deploy, n_players)
    # extract the model for the solution method
    model, solver = forces_models[solution_method]
    # run the "simulation"
    sim_data = sim_car_model(model, solver, n_players, solution_method, sim_length=5)
    # visualisation and report of data
    report = make_report(sim_data, solution_method)
    # save report
    report_file = os.path.join(output_dir, f"udgs_{n_players}_{solution_method}.html")
    report.to_html(report_file)


if __name__ == "__main__":
    args = _parse_args()
    _assert_args(args)
    main(args.generate_solver, args.to_deploy, args.n_players, args.solution_method, args.output_dir)
