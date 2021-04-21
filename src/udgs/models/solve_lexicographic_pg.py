from udgs.models.forces_utils import ForcesException
import numpy as np

from udgs.models.forces_def import params, p_idx
from udgs.models.forces_def.car_util import set_p_car


def solve_optimization(model, solver, n_players, problem, behavior, k, lex_level, next_spline_points, solver_it,
                       solver_time, solver_cost, optCost1, optCost2):
    """
        model: model settings
        solver: compiled solver
        n_players: number of players involved in the game
        problem: problem definition contains xinit, x0, all_params for the solver
        behaviour: parameters for cost function
        k: current index in simulation
        lex_level: lexicographic level
        next_spline_points: spline points for the player
        solver_it, solver_time, solver_cost: array for keeping track of info
        optCost1, optcost2: terminal cost for cumulative slack and cumulative rules
    """
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
        if exitflag == -7:
            print(f"Stalled line search at simulation step {k}")
            solver_it[k, lex_level] = info.it
            solver_time[k, lex_level] = info.solvetime
            solver_cost[k, lex_level] = info.pobj
        else:
            print(f"At simulation step {k}")
            raise ForcesException(exitflag)
    else:
        solver_it[k, lex_level] = info.it
        solver_time[k, lex_level] = info.solvetime
        solver_cost[k, lex_level] = info.pobj

    return output, problem, p_vector


def solve_lexicographic(model, solver, num_players, problem, behavior_init, behavior_first, behavior_second,
                        behavior_third, k, lexi_iter, next_spline_points, solver_it_lexi, solver_time_lexi,
                        solver_cost_lexi):

    safety_slack = 0.01
    safety_lat = 1
    if k == 0:
        output, problem, p_vector = solve_optimization(
            model, solver, num_players, problem, behavior_init, k, 0,
            next_spline_points, solver_it_lexi, solver_time_lexi,
            solver_cost_lexi, behavior_init[p_idx.OptCost1],
            behavior_init[p_idx.OptCost2])
        problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]

    for lex_level in range(lexi_iter):
        if lex_level == 0:
            output, problem, p_vector = solve_optimization(
                model, solver, num_players, problem, behavior_first,
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
                model, solver, num_players, problem, behavior_second,
                k, lex_level, next_spline_points, solver_it_lexi,
                solver_time_lexi, solver_cost_lexi,
                slackcost + safety_slack,
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
                model, solver, num_players, problem, behavior_third,
                k, lex_level, next_spline_points, solver_it_lexi,
                solver_time_lexi, solver_cost_lexi,
                slackcost + safety_slack, cumlatcost + safety_lat)

    # Extract output and initialize next iteration with current solution shifted by one stage
    problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]
    problem["x0"][model.nvar * (model.N - 1): model.nvar * model.N] = output["all_var"][model.nvar * (
            model.N - 1):model.nvar * model.N]

    temp = output["all_var"].reshape(model.nvar, model.N, order='F')
    return temp, problem, p_vector
