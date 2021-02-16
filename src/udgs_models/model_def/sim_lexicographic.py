from udgs_models.forces_utils import ForcesException
import numpy as np

from udgs_models.model_def import params
from udgs_models.model_def.car_util import set_p_car


def round_optimization(model, solver, num_cars, problem, behavior, x, k, jj, next_spline_points, solver_it, solver_time,
                      solver_cost, optCost1, optCost2):
    """
    """
    # Set initial state
    problem["xinit"] = x[:, k]
    # Set runtime parameters (the only really changing between stages are the next control points of the spline)
    p_vector = set_p_car(
        SpeedLimit=behavior.maxspeed,
        TargetSpeed=behavior.targetspeed,
        OptCost1=optCost1,
        OptCost2=optCost2,
        Xobstacle=behavior.Xobstacle,
        Yobstacle=behavior.Yobstacle,
        TargetProg=behavior.targetprog,
        kAboveTargetSpeedCost=behavior.pspeedcostA,
        kBelowTargetSpeedCost=behavior.pspeedcostB,
        kAboveSpeedLimit=behavior.pspeedcostM,
        kLag=behavior.plag,
        kLat=behavior.plat,
        pLeftLane=behavior.pLeftLane,
        kReg_dAb=behavior.pab,
        kReg_dDelta=behavior.pdotbeta,
        carLength=behavior.carLength,
        minSafetyDistance=behavior.distance,
        kSlack=behavior.pslack,
        points=next_spline_points[:, :, k],
        num_cars=num_cars
    )  # fixme check order here
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


def sim_lexicographic(model, solver, num_cars, problem, behavior_init, behavior_first, behavior_second, behavior_third,
                      x, k, next_spline_points, solver_it_lexi, solver_time_lexi, solver_cost_lexi):
    if k == 0:
        output, problem, p_vector = round_optimization(model, solver, num_cars, problem, behavior_init, x, k, 0,
                                                       next_spline_points, solver_it_lexi, solver_time_lexi,
                                                       solver_cost_lexi, behavior_init.optcost1,
                                                       behavior_init.optcost2)
        problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]

    for jj in range(3):
        if jj == 0:
            output, problem, p_vector = round_optimization(model, solver, num_cars, problem, behavior_first, x,
                                                           k, jj, next_spline_points, solver_it_lexi,
                                                           solver_time_lexi, solver_cost_lexi,
                                                           behavior_first.optcost1, behavior_first.optcost2)
            problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]
            temp = output["all_var"].reshape(model.nvar, model.N, order='F')
            row, col = temp.shape
            slackcost = 0
            for zz in range(num_cars):
                upd_s_idx = zz * params.n_states + (num_cars - 1) * params.n_inputs
                slackcost = slackcost + temp[params.s_idx.CumSlackCost + upd_s_idx, col-1]

        elif jj == 1:
            output, problem, p_vector = round_optimization(model, solver, num_cars, problem, behavior_second, x,
                                                           k, jj, next_spline_points, solver_it_lexi,
                                                           solver_time_lexi, solver_cost_lexi,
                                                           slackcost + 0.03 * slackcost,
                                                           behavior_second.optcost2)
            problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]
            temp = output["all_var"].reshape(model.nvar, model.N, order='F')
            row, col = temp.shape
            cumlatcost = 0
            for zz in range(num_cars):
                upd_s_idx = zz * params.n_states + (num_cars - 1) * params.n_inputs
                cumlatcost = cumlatcost + temp[params.s_idx.CumLatSpeedCost + upd_s_idx, col - 1]
            slackcost = 0
            for zz in range(num_cars):
                upd_s_idx = zz * params.n_states + (num_cars - 1) * params.n_inputs
                slackcost = slackcost + temp[params.s_idx.CumSlackCost + upd_s_idx, col - 1]
        else:
            output, problem, p_vector = round_optimization(model, solver, num_cars, problem, behavior_third, x,
                                                           k, jj, next_spline_points, solver_it_lexi,
                                                           solver_time_lexi, solver_cost_lexi,
                                                           slackcost + 0.1, cumlatcost + 1)

    # Extract output and initialize next iteration with current solution shifted by one stage
    problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]
    problem["x0"][model.nvar * (model.N - 1): model.nvar * model.N] = output["all_var"][model.nvar * (
                  model.N - 1):model.nvar * model.N]

    temp = output["all_var"].reshape(model.nvar, model.N, order='F')
    # row, col = temp.shape
    # cumlatcost1 = 0
    # for zz in range(num_cars):
    #     upd_s_idx = zz * params.n_states + (num_cars - 1) * params.n_inputs
    #     cumlatcost1 = cumlatcost1 + temp[params.s_idx.CumLatSpeedCost + upd_s_idx, col - 1]
    return temp, problem, p_vector
