from udgs.models import IdxParams
from udgs.models.forces_utils import ForcesException
import numpy as np

from udgs.models.forces_def import params, p_idx, x_idx, SolutionMethod, IBR, LexicographicIBR
from udgs.models.forces_def.car_util import set_p_car_ibr


def solve_optimization_br(model, solver, currentplayer, n_players, problem, behavior, optCost1, optCost2, k:int,
                          lex_level, next_spline_points, solver_it, solver_time, solver_cost,
                          playerstrajX,
                          playerstrajY,
                          max_iter: int):
    """
        model: model settings
        solver: compiled solver
        currentplayer: current player depending from order of players
        n_players: number of players involved in the game
        problem: problem definition contains xinit, x0, all_params for the solver
        behaviour: parameters for cost function
        optCost1, optcost2: terminal cost for cumulative slack and cumulative rules
        k: current index in simulation
        lex_level: lexicographic level
        next_spline_points: spline points for the player
        solver_it, solver_time, solver_cost: array for keep track of info
        playerstrajX : trajectories on X axis of each player
        playerstrajY: trajectories on Y axis of each player
        iter: number of iterations
    """
    # Set runtime parameters (the only really changing between stages are the next control points of the spline +
    # trajectories of each player)
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
    # Make sure the solver has exited properly. In event of "stalled line search", that is due to a bad initialization
    # of the solver, the procedure reinitialize the solver with some checks. The bad initialization is due to the hard
    # nonlinear constraints implemented.
    if exitflag < 0:
        if exitflag == -7:
            if lex_level == 0:
                problem['x0'][-3] = behavior[IdxParams.TargetProg] + 1
                if problem['x0'][-2] <= 0:
                    problem['x0'][-2] = 0.000001
                if problem['x0'][-1] <= 0:
                    problem['x0'][-1] = 0.000001
                output, exitflag, info = solver.solve(problem)
                if exitflag == -7:
                    problem['x0'][-3] = behavior[IdxParams.TargetProg] + 2
                    problem['x0'][-2] = 0
                    problem['x0'][-1] = 0
                    output, exitflag, info = solver.solve(problem)
                    if exitflag == -7:
                        print(
                            f"Stalled line search at simulation step {k}, agent {currentplayer + 1}, iter: {iter},"
                            f" lexlevel: {lex_level}, check solver initialization")
            elif lex_level == 1:
                problem['x0'][-3] = behavior[IdxParams.TargetProg] + 1
                if problem['x0'][-2] >= optCost1:
                    problem['x0'][-2] = optCost1
                elif problem['x0'][-2] < 0:
                    problem['x0'][-2] = 0
                if problem['x0'][-1] < 0:
                    problem['x0'][-1] = 0

                output, exitflag, info = solver.solve(problem)
                if exitflag == -7:
                    problem['x0'][-3] = behavior[IdxParams.TargetProg] + 2
                    problem['x0'][-2] = 0
                    problem['x0'][-1] = 0
                    output, exitflag, info = solver.solve(problem)
                    if exitflag == -7:
                        print(
                            f"Stalled line search at simulation step {k}, agent {currentplayer + 1}, iter: {iter},"
                            f" lexlevel: {lex_level}, check solver initialization")
            elif lex_level == 2:
                problem['x0'][-3] = behavior[IdxParams.TargetProg] + 1
                if problem['x0'][-2] > optCost1:
                    problem['x0'][-2] = optCost1
                elif problem['x0'][-2] < 0:
                    problem['x0'][-2] = 0
                if problem['x0'][-1] > optCost2:
                    problem['x0'][-1] = optCost2
                elif problem['x0'][-1] < 0:
                    problem['x0'][-1] = 0
                output, exitflag, info = solver.solve(problem)
                if exitflag == -7:
                    problem['x0'][-3] = behavior[IdxParams.TargetProg] + 2
                    problem['x0'][-2] = 0
                    problem['x0'][-1] = 0
                    output, exitflag, info = solver.solve(problem)
                    if exitflag == -7:
                        print(
                            f"Stalled line search at simulation step {k}, agent {currentplayer + 1}, iter: {iter},"
                            f" lexlevel: {lex_level}, check solver initialization")
            solver_it[k, lex_level, currentplayer, iter] = info.it
            solver_time[k, lex_level, currentplayer, iter] = info.solvetime
            solver_cost[k, lex_level, currentplayer, iter] = info.pobj
        else:
            print(f"At simulation step {k}")
            raise ForcesException(exitflag)
    else:
        solver_it[k, lex_level, currentplayer, max_iter] = info.it
        solver_time[k, lex_level, currentplayer, max_iter] = info.solvetime
        solver_cost[k, lex_level, currentplayer, max_iter] = info.pobj

    return output, problem, p_vector


# todo this function iterates best response optimization
def iterated_best_response(model, solver, order, n_players, problem_list,
                           solution_method: SolutionMethod,
                           behavior,
                           behavior_first,
                           behavior_second,
                           k, max_iter, lexi_iter, next_spline_points, solver_it, solver_time,
                           solver_cost, convergence_iter, playerstrajX, playerstrajY):
    """
        model: model settings
        solver: compiled solver
        order: order of players for IBR
        n_players: number of players involved in the game
        problem_list: list of problems, containing xinit, x0, all_params for each player
        behaviour: parameters for cost function
        behaviour_first: parameters for cost function
        behaviour_second: parameters for cost function
        k: current index in simulation
        max_iters: max number of iterations allowed
        next_spline_points: spline points for the player
        solver_it, solver_time, solver_cost: array for keep track of info
        playerstrajX : trajectories on X axis of each player
        playerstrajY: trajectories on Y axis of each player
    """
    iter = 0
    safety_slack = 0.0001  # to prevent numerical issues
    safety_lat = 0.0001  # to prevent numerical issues
    output = {}
    outputNew = np.zeros((n_players, model.nvar, model.N))
    p_vector = np.zeros((n_players, model.npar))
    playerstrajX_old = np.copy(playerstrajX)
    playerstrajY_old = np.copy(playerstrajY)
    eucl_dist = np.zeros(n_players)
    while iter < max_iter:
        for case in range(len(order)):
            if solution_method == IBR:  # normal ibr
                output[order[case]], problem_list[order[case]], p_vector[order[case], :] = \
                    solve_optimization_br(model, solver, order[case], n_players, problem_list[order[case]],
                                          behavior, behavior[p_idx.OptCost1],
                                          behavior[p_idx.OptCost2], k, 0, next_spline_points[order[case]],
                                          solver_it, solver_time, solver_cost, playerstrajX, playerstrajY, iter)
                outputNew[order[case], :, :] = output[order[case]]["all_var"].reshape(model.nvar, model.N, order='F')
                playerstrajX[order[case]] = outputNew[order[case], x_idx.X, :]
                playerstrajY[order[case]] = outputNew[order[case], x_idx.Y, :]
                problem_list[order[case]]["x0"][0: model.nvar * model.N] = \
                    output[order[case]]["all_var"][0: model.nvar * model.N]
            elif solution_method == LexicographicIBR:  # lexi ibr
                for lex_level in range(lexi_iter):
                    if lex_level == 0:
                        output[order[case]], problem_list[order[case]], p_vector[order[case], :] = \
                            solve_optimization_br(model, solver, order[case], n_players, problem_list[order[case]],
                                                  behavior_first, behavior_first[p_idx.OptCost1],
                                                  behavior_first[p_idx.OptCost2], k, lex_level,
                                                  next_spline_points[order[case]], solver_it, solver_time, solver_cost,
                                                  playerstrajX, playerstrajY, iter)
                        outputNew[order[case], :, :] = output[order[case]]["all_var"].reshape(model.nvar, model.N,
                                                                                              order='F')
                        problem_list[order[case]]["x0"][0: model.nvar * model.N] = \
                            output[order[case]]["all_var"][0: model.nvar * model.N]
                        slackcost = outputNew[order[case], :, :][params.x_idx.CumSlackCost, - 1] + safety_slack

                    elif lex_level == 1:
                        output[order[case]], problem_list[order[case]], p_vector[order[case], :] = \
                            solve_optimization_br(model, solver, order[case], n_players, problem_list[order[case]],
                                                  behavior_second, slackcost, behavior_second[p_idx.OptCost2], k,
                                                  lex_level, next_spline_points[order[case]], solver_it, solver_time,
                                                  solver_cost, playerstrajX, playerstrajY, iter)
                        outputNew[order[case], :, :] = output[order[case]]["all_var"].reshape(model.nvar, model.N,
                                                                                              order='F')
                        problem_list[order[case]]["x0"][0: model.nvar * model.N] = \
                            output[order[case]]["all_var"][0: model.nvar * model.N]
                        slackcost = outputNew[order[case], :, :][params.x_idx.CumSlackCost, - 1] + safety_slack
                        cumlatcost = outputNew[order[case], :, :][params.x_idx.CumLatSpeedCost, - 1] + safety_lat

                    else:
                        output[order[case]], problem_list[order[case]], p_vector[order[case], :] = \
                            solve_optimization_br(model, solver, order[case], n_players, problem_list[order[case]],
                                                  behavior_second, slackcost, cumlatcost, k, lex_level,
                                                  next_spline_points[order[case]], solver_it, solver_time, solver_cost,
                                                  playerstrajX, playerstrajY, iter)
                        outputNew[order[case], :, :] = output[order[case]]["all_var"].reshape(model.nvar, model.N,
                                                                                              order='F')
                        playerstrajX[order[case]] = outputNew[order[case], x_idx.X, :]
                        playerstrajY[order[case]] = outputNew[order[case], x_idx.Y, :]
                        problem_list[order[case]]["x0"][0: model.nvar * model.N] = \
                            output[order[case]]["all_var"][0: model.nvar * model.N]
            else:
                raise ValueError(f"{solution_method} is not supported")

        # verify convergence
        for i in range(n_players):
            eucl_dist[i] = np.sum(np.sqrt(np.square(playerstrajX[i] - playerstrajX_old[i]) +
                                          np.square(playerstrajY[i] - playerstrajY_old[i])))

        if all(i <= sim_params.ibr_convergence_thresh for i in eucl_dist):
            print(f"IBR: iterations required for convergence: {iter}")
            convergence_iter[k] = iter
            return output, problem_list, p_vector
        else:
            playerstrajX_old = np.copy(playerstrajX)
            playerstrajY_old = np.copy(playerstrajY)

        iter += 1
    print(f"IBR - convergence not reached after {iter} iterations of players' updates")
    convergence_iter[k] = iter + 1
    return output, problem_list, p_vector
