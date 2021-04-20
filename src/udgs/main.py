import argparse
import os.path

from udgs.forces_models.model_def.sim import sim_car_model
from udgs.forces_models.model_def.generate_model import AVAILABLE_METHODS, generate_forces_models, SolutionMethod, \
    LexicographicPG
from udgs.forces_models.model_def.sim_report import make_report


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
        help="If True generates the solver with additional flags for targeting Docker Linux platform (gokart)",
        type=bool,
    )
    p.add_argument(
        "--num_cars",
        default=3,
        help="todo",
        type=int,
    )
    p.add_argument(
        "--output_dir",
        default="out-docker",
        help="todo",
        type=str,
    )
    p.add_argument(
        "--solution_method",
        default="LexicographicPG",
        help="PG: potential game solution,"
             "LexicographicPG: Lexicographic potential game"
             "IBR: Iterated best response"
             "LexicographicIBR: Lexicographic Iterated best response",

        type=str,
    )
    return p.parse_args()


def main(generate_solver: bool = True,
         to_deploy: bool = False,
         n_players: int = 3,
         solution_method: SolutionMethod = LexicographicPG,
         output_dir: str = "out-docker"):
    if solution_method not in AVAILABLE_METHODS:
        raise ValueError(f"{solution_method} is not available. Retry with one amongst {AVAILABLE_METHODS}")
    # generate forces models definition and solvers
    forces_models = generate_forces_models(generate_solver, to_deploy, n_players)
    # extract the model for the solution method
    model, solver = forces_models[solution_method]
    # run the "simulation"
    sim_data = sim_car_model(model, solver, n_players, solution_method, sim_length=5)
    # visualisation and report of data
    report = make_report(sim_data, solution_method)
    # save report
    report_file = os.path.join(output_dir, "udgs_report.html")
    report.to_html(report_file)


if __name__ == "__main__":
    args = _parse_args()
    main(args.generate_solver, args.to_deploy, args.num_cars, args.solution_method)
