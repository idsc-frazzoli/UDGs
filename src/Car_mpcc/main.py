import argparse
import forcespro

from Car_mpcc.Car_optimizer.sim import sim_car_model
from Car_mpcc.Car_optimizer.sim_report import make_report
from Car_optimizer.generate_model import generate_car_model
from tracks import straightLineR2L, winti_002, straightLineN2S


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mpc_model", default="human-constraints", help="todo", type=str)
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
        default=2,
        help="If True generates the solver with additional flags for targeting Docker Linux platform (gokart)",
        type=int,
    )
    return p.parse_args()


def _generate_model(mpc_model: str, generate_solver: bool = True, to_deploy: bool = False, num_cars: int = 2):
    if mpc_model == "human-constraints":
        model, solver = generate_car_model(generate_solver, to_deploy, num_cars)
        sim_data = sim_car_model(model, solver, num_cars, sim_length=10, track=straightLineR2L, track2=straightLineN2S)
        make_report(sim_data, num_cars)
    else:
        raise ValueError(f'The requested model "{mpc_model}" is not recognized.')


if __name__ == "__main__":
    args = _parse_args()
    _generate_model(args.mpc_model, args.generate_solver, args.to_deploy)
