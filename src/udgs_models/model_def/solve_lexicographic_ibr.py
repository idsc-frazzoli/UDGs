from udgs_models.forces_utils import ForcesException
import numpy as np

from udgs_models.model_def import params, p_idx, x_idx
from udgs_models.model_def.car_util import set_p_car_ibr


def solve_optimization_br(model, solver, currentplayer, n_players, problem, behavior, k, jj, next_spline_points,
                          solver_it, solver_time, solver_cost, optCost1, optCost2, playerstrajX, playerstrajY):
    """

    """
    p_vector = set_p_car_ibr(
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
        coordinateX_car=0,
        coordinateY_car=0,
        n_players=n_players
    )
    problem["all_parameters"] = np.tile(p_vector, model.N)
    jj = 0  # index that updates only when i is not equal to current player index
    for i in range(n_players):
        if i == currentplayer:
            continue

        problem["all_parameters"][params.n_opt_param + 2 * jj:
                                  len(problem["all_parameters"]):model.npar] = playerstrajX[i]
        problem["all_parameters"][params.n_opt_param + 1 + 2 * jj:
                                  len(problem["all_parameters"]):model.npar] = playerstrajY[i]
        jj += 1

    # Time to solve the NLP!
    output, exitflag, info = solver.solve(problem)
    # Make sure the solver has exited properly.
    temp = output["all_var"].reshape(model.nvar, model.N, order='F')
    if exitflag < 0:
        print(f"At simulation step {k}")
        raise ForcesException(exitflag)
    else:
        solver_it[k, jj] = info.it
        solver_time[k, jj] = info.solvetime
        solver_cost[k, jj] = info.pobj

    return output, problem, p_vector


# todo this function iterates best response optimization
def iterated_best_response(model, solver, order, n_players, problem_list, behavior, k, jj, next_spline_points,
                           solver_it, solver_time, solver_cost, optCost1, optCost2, outputOld, playerstrajX,
                           playerstrajY):
    """
    """
    iter = 0
    output = {}
    outputNew = np.zeros((n_players, model.nvar, model.N))
    p_vector = np.zeros((n_players, model.npar))
    playerstrajX_old = np.copy(playerstrajX)
    playerstrajY_old = np.copy(playerstrajY)
    eucl_dist = np.zeros((n_players))
    while iter <= 10:
        iter += 1
        for case in range(len(order)):
                output[order[case]], problem_list[order[case]], p_vector[order[case], :] =\
                    solve_optimization_br(model, solver, case, n_players, problem_list[order[case]], behavior, k, 0,
                                          next_spline_points[order[case]], solver_it, solver_time, solver_cost,
                                          optCost1, optCost2, playerstrajX[order[case]], playerstrajY[order[case]])
                outputNew[order[case], :, :] = output[order[case]]["all_var"].reshape(model.nvar, model.N, order='F')
                playerstrajX[order[case]] = outputNew[order[case], x_idx.X, :]
                playerstrajY[order[case]] = outputNew[order[case], x_idx.Y, :]
                problem_list[order[case]]["x0"][0: model.nvar * (model.N - 1)] =\
                    output[order[case]]["all_var"][model.nvar: model.nvar * model.N]

        # verify convergence
        for i in range(n_players):
                eucl_dist[i] = np.sum(np.sqrt(np.square(playerstrajX[i] - playerstrajX_old[i]) +
                                       np.square(playerstrajY[i] - playerstrajY_old[i])))

        if all(i <= 0.1 for i in eucl_dist):
            return output, problem_list, p_vector
        else:
            playerstrajX_old = np.copy(playerstrajX)
            playerstrajY_old = np.copy(playerstrajY)

    return output, problem_list, p_vector


# todo call three times best response ibr
def solve_lexicographic_ibr(model, solver, num_players, problem, behavior_init, behavior_first, behavior_second,
                        behavior_third, x, k, next_spline_points, solver_it_lexi, solver_time_lexi, solver_cost_lexi):
    if k == 0:
        output, problem, p_vector = solve_optimization_br(
            model, solver, num_players, problem, behavior_init, x, k, 0,
            next_spline_points, solver_it_lexi, solver_time_lexi,
            solver_cost_lexi, behavior_init[p_idx.OptCost1],
            behavior_init[p_idx.OptCost2])
        problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]

    for lex_level in range(3):
        if lex_level == 0:
            output, problem, p_vector = solve_optimization_br(
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
            output, problem, p_vector = solve_optimization_br(
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
            output, problem, p_vector = solve_optimization_br(
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
