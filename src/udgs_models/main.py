import argparse
import forcespro

from udgs_models.model_def.sim import sim_car_model
from udgs_models.model_def.sim_report import make_report
from model_def.generate_model import generate_car_model


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mpc_model", default="human-constraints", help="todo", type=str)
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
        "--condition",
        default=2,
        help="0: PG, 1:LexiPG, 2:IBR, 3:LexiIBR",
        type=int,
    )
    return p.parse_args()


def _generate_model(mpc_model: str, generate_solver: bool = True, to_deploy: bool = False, num_cars: int = 3,
                    condition: int = 2):
    if mpc_model == "human-constraints":
        model, solver = generate_car_model(generate_solver, to_deploy, num_cars, condition)
        sim_data = sim_car_model(model, solver, num_cars, condition, sim_length=45)
        make_report(sim_data)
    else:
        raise ValueError(f'The requested model "{mpc_model}" is not recognized.')


if __name__ == "__main__":
    args = _parse_args()
    _generate_model(args.mpc_model, args.generate_solver, args.to_deploy, args.num_cars, args.condition)
