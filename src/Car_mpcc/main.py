import argparse
import forcespro

from Car_mpcc.Car_optimizer.sim import sim_car_model
from Car_mpcc.Car_optimizer.sim_report import make_report
from Car_optimizer.generate_model import generate_car_model
from tracks import straightLineR2L, winti_002


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
    return p.parse_args()


def _generate_model(mpc_model: str, generate_solver: bool = True, to_deploy: bool = False):
    if mpc_model == "human-constraints":
        model, solver = generate_car_model(generate_solver, to_deploy)
        sim_data = sim_car_model(model, solver, sim_length=5, track=straightLineR2L)
        make_report(sim_data)
    else:
        raise ValueError(f'The requested model "{mpc_model}" is not recognized.')


if __name__ == "__main__":
    args = _parse_args()
    _generate_model(args.mpc_model, args.generate_solver, args.to_deploy)
