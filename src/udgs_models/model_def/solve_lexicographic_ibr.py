from udgs_models.forces_utils import ForcesException
import numpy as np

from udgs_models.model_def import params, p_idx, x_idx
from udgs_models.model_def.car_util import set_p_car_ibr


def solve_optimization_br(model, solver, currentplayer, n_players, problem, behavior, optCost1, optCost2, k, jj,
                          lex_level, next_spline_points, solver_it, solver_time, solver_cost, playerstrajX,
                          playerstrajY):
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
    kk = 0  # index that updates only when i is not equal to current player index
    for i in range(n_players):
        if i == currentplayer:
            continue

        problem["all_parameters"][params.n_opt_param + 3 * params.n_bspline_points + 2 * kk:
                                  len(problem["all_parameters"]):model.npar] = playerstrajX[i]
        problem["all_parameters"][params.n_opt_param + 3 * params.n_bspline_points + 1 + 2 * kk:
                                  len(problem["all_parameters"]):model.npar] = playerstrajY[i]
        kk += 1

    # Time to solve the NLP!
    output, exitflag, info = solver.solve(problem)
    # Make sure the solver has exited properly.
    temp = output["all_var"].reshape(model.nvar, model.N, order='F')
    if exitflag < 0:

        if exitflag == -7:
            print(f"Stalled line search at simulation step {k}, agent {currentplayer+1}")
        else:
            print(f"At simulation step {k}")
            raise ForcesException(exitflag)
    else:
        solver_it[k, jj, lex_level] = info.it
        solver_time[k, jj, lex_level] = info.solvetime
        solver_cost[k, jj, lex_level] = info.pobj

    return output, problem, p_vector


# todo this function iterates best response optimization
def iterated_best_response(model, solver, order, n_players, problem_list, condition, behavior, behavior_first,
                           behavior_second, k, next_spline_points, solver_it, solver_time, solver_cost,
                           playerstrajX, playerstrajY):
    """
    """
    iter = 0
    output = {}
    outputNew = np.zeros((n_players, model.nvar, model.N))
    p_vector = np.zeros((n_players, model.npar))
    playerstrajX_old = np.copy(playerstrajX)
    playerstrajY_old = np.copy(playerstrajY)
    eucl_dist = np.zeros(n_players)
    while iter <= 10:
        iter += 1
        for case in range(len(order)):
            if condition == 2:  # normal ibr
                output[order[case]], problem_list[order[case]], p_vector[order[case], :] =\
                    solve_optimization_br(model, solver, case, n_players, problem_list[order[case]],
                                          behavior, behavior[p_idx.OptCost1],
                                          behavior[p_idx.OptCost2], k, order[case], 0, next_spline_points[order[case]],
                                          solver_it, solver_time, solver_cost, playerstrajX, playerstrajY)
                outputNew[order[case], :, :] = output[order[case]]["all_var"].reshape(model.nvar, model.N, order='F')
                playerstrajX[order[case]] = outputNew[order[case], x_idx.X, :]
                playerstrajY[order[case]] = outputNew[order[case], x_idx.Y, :]
                problem_list[order[case]]["x0"][0: model.nvar * model.N] =\
                    output[order[case]]["all_var"][0: model.nvar * model.N]
            else:  # lexi ibr
                for lex_level in range(3):
                    if lex_level == 0:
                        output[order[case]], problem_list[order[case]], p_vector[order[case], :] = \
                            solve_optimization_br(model, solver, case, n_players, problem_list[order[case]],
                                                  behavior_first, behavior_first[p_idx.OptCost1],
                                                  behavior_first[p_idx.OptCost2], k, order[case], lex_level,
                                                  next_spline_points[order[case]], solver_it, solver_time, solver_cost,
                                                  playerstrajX, playerstrajY)
                        outputNew[order[case], :, :] = output[order[case]]["all_var"].reshape(model.nvar, model.N,
                                                                                              order='F')
                        # playerstrajX[order[case]] = outputNew[order[case], x_idx.X, :]
                        # playerstrajY[order[case]] = outputNew[order[case], x_idx.Y, :]
                        problem_list[order[case]]["x0"][0: model.nvar * model.N] = \
                            output[order[case]]["all_var"][0: model.nvar * model.N]
                        slackcost = outputNew[order[case], :, :][params.x_idx.CumSlackCost, - 1]

                    elif lex_level == 1:
                        output[order[case]], problem_list[order[case]], p_vector[order[case], :] = \
                            solve_optimization_br(model, solver, case, n_players, problem_list[order[case]],
                                                  behavior_second, slackcost,
                                                  behavior_second[p_idx.OptCost2], k, order[case], lex_level,
                                                  next_spline_points[order[case]], solver_it, solver_time, solver_cost,
                                                  playerstrajX, playerstrajY)
                        outputNew[order[case], :, :] = output[order[case]]["all_var"].reshape(model.nvar, model.N,
                                                                                              order='F')
                        # playerstrajX[order[case]] = outputNew[order[case], x_idx.X, :]
                        # playerstrajY[order[case]] = outputNew[order[case], x_idx.Y, :]
                        problem_list[order[case]]["x0"][0: model.nvar * model.N] = \
                            output[order[case]]["all_var"][0: model.nvar * model.N]
                        slackcost = outputNew[order[case], :, :][params.x_idx.CumSlackCost, - 1]
                        cumlatcost = outputNew[order[case], :, :][params.x_idx.CumLatSpeedCost, - 1] + 0.1

                    else:
                        output[order[case]], problem_list[order[case]], p_vector[order[case], :] = \
                            solve_optimization_br(model, solver, case, n_players, problem_list[order[case]],
                                                  behavior_second, slackcost, cumlatcost, k, order[case], lex_level,
                                                  next_spline_points[order[case]], solver_it, solver_time, solver_cost,
                                                  playerstrajX, playerstrajY)
                        outputNew[order[case], :, :] = output[order[case]]["all_var"].reshape(model.nvar, model.N,
                                                                                              order='F')
                        playerstrajX[order[case]] = outputNew[order[case], x_idx.X, :]
                        playerstrajY[order[case]] = outputNew[order[case], x_idx.Y, :]
                        problem_list[order[case]]["x0"][0: model.nvar * model.N] = \
                            output[order[case]]["all_var"][0: model.nvar * model.N]

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
