import argparse

import forcespro

from udgs_models.model_def.sim import sim_car_model
from udgs_models.model_def.sim_report import make_report
from model_def.generate_model import _generate_forces_model, AVAILABLE_METHODS, generate_forces_models, SolutionMethod, \
    LexicographicPG


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--generate_solver",
        default=True,
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
        "--solution_method",
        default="LexicographicPG",
        help="PG, LexicographicPG,IBR,LexicographicIBR",
        type=str,
    )
    return p.parse_args()


def main(generate_solver: bool = True,
         to_deploy: bool = False,
         n_players: int = 3,
         solution_method: SolutionMethod = LexicographicPG):
    assert solution_method in AVAILABLE_METHODS, solution_method
    forces_models = generate_forces_models(generate_solver, to_deploy, n_players)
    # extract the model for the solution method
    model, solver = forces_models[solution_method]
    sim_data = sim_car_model(model, solver, n_players, solution_method, sim_length=2)
    make_report(sim_data, solution_method)


if __name__ == "__main__":
    args = _parse_args()
    main(args.generate_solver, args.to_deploy, args.num_cars, args.solution_method)
