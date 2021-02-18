from udgs_models.forces_utils import ForcesException
import numpy as np

from udgs_models.model_def import params, p_idx
from udgs_models.model_def.car_util import set_p_car


def solve_optimization(model, solver, n_players, problem, behavior, x, k, jj, next_spline_points, solver_it,
                       solver_time, solver_cost, optCost1, optCost2):
    """
    """
    # Set initial state
    problem["xinit"] = x[:, k]
    # Set runtime parameters (the only really changing between stages are the next control points of the spline)
    p_vector = set_p_car(
        SpeedLimit=behavior[p_idx.SpeedLimit],
        TargetSpeed=behavior[p_idx.TargetSpeed],
        OptCost1=optCost1,
        OptCost2=optCost2,
        Xobstacle=behavior[p_idx.Xobstacle],
        Yobstacle=behavior[p_idx.Yobstacle],
        TargetProg=behavior[p_idx.TargetProg],
        kAboveTargetSpeedCost=behavior[p_idx.kAboveTargetSpeedCost],
        kBelowTargetSpeedCost=behavior[p_idx.kBelowTargetSpeedCost],
        kAboveSpeedLimit=behavior[p_idx.kAboveSpeedLimit],
        kLag=behavior[p_idx.kLag],
        kLat=behavior[p_idx.kLat],
        pLeftLane=behavior[p_idx.pLeftLane],
        kReg_dAb=behavior[p_idx.kReg_dAb],
        kReg_dDelta=behavior[p_idx.kReg_dDelta],
        carLength=behavior[p_idx.carLength],
        minSafetyDistance=behavior[p_idx.minSafetyDistance],
        kSlack=behavior[p_idx.kSlack],
        points=next_spline_points[:, :, k],
        n_players=n_players
    )
    problem["all_parameters"] = np.tile(p_vector, (model.N,))

    # Time to solve the NLP!
    output, exitflag, info = solver.solve(problem)
    # Make sure the solver has exited properly.
    if exitflag < 0:
        print(f"At simulation step {k}")
        raise ForcesException(exitflag)
    else:
        solver_it[k, jj] = info.it
        solver_time[k, jj] = info.solvetime
        solver_cost[k, jj] = info.pobj

    return output, problem, p_vector


def solve_lexicographic(model, solver, num_players, problem,
                        behavior_init,
                        behavior_first,
                        behavior_second,
                        behavior_third,
                        x, k, next_spline_points, solver_it_lexi, solver_time_lexi, solver_cost_lexi):
    if k == 0:
        output, problem, p_vector = solve_optimization(
            model, solver, num_players, problem, behavior_init, x, k, 0,
            next_spline_points, solver_it_lexi, solver_time_lexi,
            solver_cost_lexi, behavior_init[p_idx.OptCost1],
            behavior_init[p_idx.OptCost2])
        problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]

    for lex_level in range(3):
        if lex_level == 0:
            output, problem, p_vector = solve_optimization(
                model, solver, num_players, problem, behavior_first, x,
                k, lex_level, next_spline_points, solver_it_lexi,
                solver_time_lexi, solver_cost_lexi,
                behavior_first[p_idx.OptCost1],
                behavior_first[p_idx.OptCost2])
            problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]
            temp = output["all_var"].reshape(model.nvar, model.N, order='F')
            row, col = temp.shape
            slackcost = 0
            for zz in range(num_players):
                upd_s_idx = zz * params.n_states + (num_players - 1) * params.n_inputs
                slackcost = slackcost + temp[params.x_idx.CumSlackCost + upd_s_idx, col - 1]

        elif lex_level == 1:
            output, problem, p_vector = solve_optimization(
                model, solver, num_players, problem, behavior_second, x,
                k, lex_level, next_spline_points, solver_it_lexi,
                solver_time_lexi, solver_cost_lexi,
                slackcost + 0.03 * slackcost,
                behavior_second[p_idx.OptCost2])
            problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]
            temp = output["all_var"].reshape(model.nvar, model.N, order='F')
            row, col = temp.shape
            cumlatcost = 0
            for zz in range(num_players):
                upd_s_idx = zz * params.n_states + (num_players - 1) * params.n_inputs
                cumlatcost = cumlatcost + temp[params.x_idx.CumLatSpeedCost + upd_s_idx, col - 1]
            slackcost = 0
            for zz in range(num_players):
                upd_s_idx = zz * params.n_states + (num_players - 1) * params.n_inputs
                slackcost = slackcost + temp[params.x_idx.CumSlackCost + upd_s_idx, col - 1]
        else:
            output, problem, p_vector = solve_optimization(
                model, solver, num_players, problem, behavior_third, x,
                k, lex_level, next_spline_points, solver_it_lexi,
                solver_time_lexi, solver_cost_lexi,
                slackcost + 0.1, cumlatcost + 1)

    # Extract output and initialize next iteration with current solution shifted by one stage
    problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]
    problem["x0"][model.nvar * (model.N - 1): model.nvar * model.N] = output["all_var"][model.nvar * (
            model.N - 1):model.nvar * model.N]

    temp = output["all_var"].reshape(model.nvar, model.N, order='F')
    return temp, problem, p_vector
