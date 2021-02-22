from dataclasses import dataclass
from typing import Tuple, Mapping

from scipy.integrate import solve_ivp

from bspline.bspline import casadiDynamicBSPLINE, atan2, casadiDynamicBSPLINEforward, casadiDynamicBSPLINEsidewards
from udgs_models.model_def import params, p_idx
import numpy as np
import itertools

from udgs_models.model_def.driver_config import behaviors_zoo
from tracks import Track

from tracks.zoo import straightLineR2L, straightLineN2S, straightLineL2R
from udgs_models.model_def.dynamics_car import dynamics_cars
from udgs_models.model_def.solve_lexicographic_ibr import solve_optimization_br, iterated_best_response
from udgs_models.model_def.solve_lexicographic_pg import solve_lexicographic, solve_optimization


@dataclass(frozen=True)
class SimPlayer:
    x: np.ndarray
    x_pred: np.ndarray
    u: np.ndarray
    u_pred: np.ndarray
    next_spline_points: np.ndarray
    track: Track


@dataclass(frozen=True)
class SimData:
    players: Mapping[int, SimPlayer]
    solver_it: np.ndarray
    solver_time: np.ndarray
    solver_cost: np.ndarray


def sim_car_model(
        model, solver, n_players, condition, sim_length: int = 200, seed: int = 1,
        tracks: Tuple[Track] = (straightLineL2R,
                                straightLineN2S,
                                straightLineR2L,
                                straightLineR2L,
                                straightLineR2L)
) -> SimData:
    """

    :param seed:
    :param tracks:
    :param model:
    :param solver:
    :param sim_length:
    :return:
    """
    # Load some parameters

    behavior_init = behaviors_zoo["initConfig"].config
    behavior_first = behaviors_zoo["firstOptim"].config
    behavior_second = behaviors_zoo["secondOptim"].config
    behavior_third = behaviors_zoo["thirdOptim"].config
    behavior_pg = behaviors_zoo["PG"].config
    behavior_ibr = behaviors_zoo["PG"].config

    n_states = params.n_states
    n_inputs = params.n_inputs
    x_idx = params.x_idx
    sim_data_players = []

    if condition == 0 or condition == 1:
        # Variables for storing simulation data
        x = np.zeros((n_states * n_players, sim_length + 1))  # states
        u = np.zeros((n_inputs * n_players, sim_length))  # inputs
        x_pred = np.zeros((n_states * n_players, model.N, sim_length + 1))  # states
        u_pred = np.zeros((n_inputs * n_players, model.N, sim_length))  # inputs
        next_spline_points = np.zeros((params.n_bspline_points * n_players, 3, sim_length))

        if condition == 0:
            solver_it = np.zeros((sim_length, 1))
            solver_time = np.zeros((sim_length, 1))
            solver_cost = np.zeros((sim_length, 1))
        else:
            solver_it = np.zeros((sim_length, 3))
            solver_time = np.zeros((sim_length, 3))
            solver_cost = np.zeros((sim_length, 3))
        # Set initial condition

        init_progress = 0.01
        x_pos = np.zeros(n_players)
        y_pos = np.zeros(n_players)
        dx = np.zeros(n_players)
        dy = np.zeros(n_players)
        dx_s = np.zeros(n_players)
        dy_s = np.zeros(n_players)
        theta_pos = np.zeros(n_players)
        xinit = np.zeros(n_states * n_players)

        for i in range(n_players):
            x_pos[i], y_pos[i] = casadiDynamicBSPLINE(init_progress, tracks[i].spline.as_np_array())
            dx[i], dy[i] = casadiDynamicBSPLINEforward(init_progress, tracks[i].spline.as_np_array())
            dx_s[i], dy_s[i] = casadiDynamicBSPLINEsidewards(init_progress, tracks[i].spline.as_np_array())
            theta_pos[i] = atan2(dy[i], dx[i])
            x_pos[i] = x_pos[i] + 1.75 * dx_s[i]
            y_pos[i] = y_pos[i] + 1.75 * dy_s[i]
            upd_s_idx = - n_inputs + i * n_states
            xinit[x_idx.X + upd_s_idx] = x_pos[i]
            xinit[x_idx.Y + upd_s_idx] = y_pos[i]
            xinit[x_idx.Theta + upd_s_idx] = theta_pos[i]
            xinit[x_idx.Vx + upd_s_idx] = 8.3

            # totally arbitrary
            xinit[x_idx.Delta + upd_s_idx] = 0  # totally arbitrary
            xinit[x_idx.S + upd_s_idx] = init_progress

        x[:, 0] = xinit
        problem = {}
        initialization = np.tile(np.append(np.zeros(n_inputs * n_players), xinit), model.N)
        problem["x0"] = initialization
        spline_start_idx = np.zeros(n_players)
        for k in range(sim_length):

            # find bspline
            # while x[x_idx.s - n_inputs, k] >= 1:
            for i in range(n_players):
                upd_s_idx = - n_inputs + i * n_states
                while x[x_idx.S + upd_s_idx, k] >= 1:
                    # spline step forward
                    spline_start_idx[i] += 1
                    # fixme some module operation (number of control points) is probably needed
                    x[x_idx.S + upd_s_idx, k] -= 1
                for j in range(params.n_bspline_points):
                    next_point = tracks[i].spline.get_control_point(spline_start_idx[i].astype(int) + j)
                    next_spline_points[j + i * params.n_bspline_points, :, k] = next_point
                # Limit acceleration
                x[x_idx.Acc + upd_s_idx, k] = min(2, x[x_idx.Acc + upd_s_idx, k])

            if condition == 0:  # PG only
                # Set initial state
                problem["xinit"] = x[:, k]

                output, problem, p_vector = solve_optimization(
                    model, solver, n_players, problem, behavior_pg, k, 0,
                    next_spline_points, solver_it, solver_time,
                    solver_cost, behavior_pg[p_idx.OptCost1], behavior_pg[p_idx.OptCost1])
                # Extract output and initialize next iteration with current solution shifted by one stage
                problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]
                problem["x0"][model.nvar * (model.N - 1): model.nvar * model.N] = output["all_var"][model.nvar * (
                        model.N - 1):model.nvar * model.N]
                temp = output["all_var"].reshape(model.nvar, model.N, order='F')

                u_pred[:, :, k] = temp[0:n_inputs * n_players, :]  # predicted inputs
                x_pred[:, :, k] = temp[n_inputs * n_players: params.n_var * n_players, :]  # predicted states

                # Apply optimized input u of first stage to system and save simulation data
                u[:, k] = u_pred[:, 0, k]

            elif condition == 1:  # PG lexicographic
                temp, problem, p_vector = solve_lexicographic(
                    model, solver, n_players, problem,
                    behavior_init,
                    behavior_first,
                    behavior_second,
                    behavior_third, x, k, next_spline_points,
                    solver_it, solver_time, solver_cost)

                u_pred[:, :, k] = temp[0:n_inputs * n_players, :]  # predicted inputs
                x_pred[:, :, k] = temp[n_inputs * n_players: params.n_var * n_players, :]  # predicted states

                # Apply optimized input u of first stage to system and save simulation data
                u[:, k] = u_pred[:, 0, k]

            # "simulate" state evolution
            sol = solve_ivp(lambda t, states: dynamics_cars[n_players](states, u[:, k], p_vector),
                            [0, params.dt_integrator_step],
                            x[:, k])
            x[:, k + 1] = sol.y[:, -1]

        # extract trajectories and values from simulation for each player

        for i in range(n_players):
            upd_s_idx = (i + 1) * params.n_states
            upd_i_idx = (i + 1) * params.n_inputs
            upd_i_spline = (i + 1) * params.n_bspline_points
            sim_data_players.append(SimPlayer(
                x=x[range(i * params.n_states, upd_s_idx), :],
                x_pred=x_pred[range(i * params.n_states, upd_s_idx), :],
                u=u[range(i * params.n_inputs, upd_i_idx), :],
                u_pred=u_pred[range(i * params.n_inputs, upd_i_idx), :],
                next_spline_points=next_spline_points[range(i * params.n_bspline_points, upd_i_spline), :],
                track=tracks[i]))

    else:  # IBR and Lexicographic IBR
        # todo find a way to store multiple players information
        # Variables for storing simulation data
        x = np.zeros((n_players, n_states, sim_length + 1))  # states
        u = np.zeros((n_players, n_inputs, sim_length))  # inputs
        x_pred = np.zeros((n_players, n_states, model.N, sim_length + 1))  # states
        u_pred = np.zeros((n_players, n_inputs, model.N, sim_length))  # inputs
        next_spline_points = np.zeros((n_players, params.n_bspline_points, 3, sim_length))

        if condition == 2:
            solver_it = np.zeros((sim_length, n_players,  1))
            solver_time = np.zeros((sim_length, n_players,  1))
            solver_cost = np.zeros((sim_length, n_players,  1))
        else:
            solver_it = np.zeros((sim_length, n_players,  3))
            solver_time = np.zeros((sim_length, n_players, 3))
            solver_cost = np.zeros((sim_length, n_players,  3))
        # Set initial condition

        init_progress = 0.01
        x_pos = np.zeros(n_players)
        y_pos = np.zeros(n_players)
        dx = np.zeros(n_players)
        dy = np.zeros(n_players)
        dx_s = np.zeros(n_players)
        dy_s = np.zeros(n_players)
        theta_pos = np.zeros(n_players)
        xinit = np.zeros((n_players, n_states))

        playerstrajX = np.zeros((n_players, model.N))
        playerstrajY = np.zeros((n_players, model.N))
        outputOld = np.zeros((n_players, n_states + n_inputs, model.N))
        output = {}
        p_vector = np.zeros((n_players, model.npar))
        for i in range(n_players):  # todo do proper initialization for each player
            x_pos[i], y_pos[i] = casadiDynamicBSPLINE(init_progress, tracks[i].spline.as_np_array())
            dx[i], dy[i] = casadiDynamicBSPLINEforward(init_progress, tracks[i].spline.as_np_array())
            dx_s[i], dy_s[i] = casadiDynamicBSPLINEsidewards(init_progress, tracks[i].spline.as_np_array())
            theta_pos[i] = atan2(dy[i], dx[i])

            theta_pos[i] = atan2(dy[i], dx[i])
            x_pos[i] = x_pos[i] + 1.75 * dx_s[i]  # move player to the right lane
            y_pos[i] = y_pos[i] + 1.75 * dy_s[i]  # move player to the right lane
            xinit[i, x_idx.X - n_inputs] = x_pos[i]
            xinit[i, x_idx.Y - n_inputs] = y_pos[i]
            xinit[i, x_idx.Theta - n_inputs] = theta_pos[i]
            xinit[i, x_idx.Vx - n_inputs] = 8.3

            # totally arbitrary
            xinit[i, x_idx.Delta - n_inputs] = 0  # totally arbitrary
            xinit[i, x_idx.S - n_inputs] = init_progress

        for i in range(n_players):
            x[i, :, 0] = xinit[i]

        problem = {}
        initialization = np.tile(np.append(np.zeros(n_inputs), xinit[0]), model.N)
        problem["x0"] = initialization
        problem_list = [problem]

        for i in range(1, n_players):
            new = problem.copy()
            initialization = np.tile(np.append(np.zeros(n_inputs), xinit[i]), model.N)
            new["x0"] = initialization
            problem_list.append(new)

        playerorderlist = list(itertools.permutations(range(0, n_players)))
        # todo implement a loop that considers all orders of players
        spline_start_idx = np.zeros(n_players)
        for k in range(sim_length):
            # find bspline
            # while x[x_idx.s - n_inputs, k] >= 1:
            for i in range(n_players):
                upd_s_idx = - n_inputs
                while x[i, x_idx.S + upd_s_idx, k] >= 1:
                    # spline step forward
                    spline_start_idx[i] += 1
                    # fixme some module operation (number of control points) is probably needed
                    x[i, x_idx.S + upd_s_idx, k] -= 1
                for j in range(params.n_bspline_points):
                    next_point = tracks[i].spline.get_control_point(spline_start_idx[i].astype(int) + j)
                    next_spline_points[i, j, :, k] = next_point
                # Limit acceleration
                x[i, x_idx.Acc + upd_s_idx, k] = min(2, x[i, x_idx.Acc + upd_s_idx, k])

            for i in range(n_players):
                problem_list[i]["xinit"] = x[i, :, k]
                output[i], problem_list[i], p_vector[i, :] = \
                    solve_optimization_br(model, solver, i, n_players, problem_list[i], behavior_init,
                                          behavior_init[p_idx.OptCost1], behavior_init[p_idx.OptCost2],
                                          k, i, 0, next_spline_points[i], solver_it,
                                          solver_time, solver_cost, playerstrajX, playerstrajY)
                outputOld[i, :, :] = output[i]["all_var"].reshape(model.nvar, model.N, order='F')
                playerstrajX[i] = outputOld[i, x_idx.X, :]
                playerstrajY[i] = outputOld[i, x_idx.Y, :]
                problem_list[i]["x0"][0: model.nvar * (model.N - 1)] = output[i]["all_var"][model.nvar:
                                                                                            model.nvar * model.N]
                problem_list[i]["x0"][model.nvar * (model.N - 1): model.nvar * model.N] = output[i]["all_var"][
                                                                                          model.nvar * (model.N - 1)
                                                                                          :model.nvar * model.N]

            output, problem, p_vector =\
                iterated_best_response(model, solver, playerorderlist[0], n_players, problem_list, condition,
                                       behavior_ibr, behavior_first, behavior_second, k,
                                       next_spline_points, solver_it, solver_time, solver_cost,
                                       playerstrajX, playerstrajY)
            # Extract output and initialize next iteration with current solution shifted by one stage

            for i in range(n_players):
                problem_list[i]["x0"][0: model.nvar * (model.N - 1)] = output[i]["all_var"][model.nvar:
                                                                                            model.nvar * model.N]
                problem_list[i]["x0"][model.nvar * (model.N - 1): model.nvar * model.N] = output[i]["all_var"][
                                                                                          model.nvar * (model.N - 1)
                                                                                          :model.nvar * model.N]
                temp = output[i]["all_var"].reshape(model.nvar, model.N, order='F')
                u_pred[i, :, :, k] = temp[0:n_inputs, :]  # predicted inputs
                x_pred[i, :, :, k] = temp[n_inputs: params.n_var, :]  # predicted states

                # Apply optimized input u of first stage to system and save simulation data
                u[i, :, k] = u_pred[i, :, 0, k]

            for i in range(n_players):
                # "simulate" state evolution
                sol = solve_ivp(lambda t, states: dynamics_cars[1](states, u[i, :, k], p_vector[i]),
                                [0, params.dt_integrator_step],
                                x[i, :, k])
                x[i, :, k + 1] = sol.y[:, -1]

        # extract trajectories and values from simulation for each player
        for i in range(n_players):
            sim_data_players.append(SimPlayer(
                x=x[i],
                x_pred=x_pred[i],
                u=u[i],
                u_pred=u_pred[i],
                next_spline_points=next_spline_points[i],
                track=tracks[i]))

    return SimData(
        players=dict(zip(range(n_players), sim_data_players)),
        solver_it=solver_it,
        solver_time=solver_time,
        solver_cost=solver_cost,
    )
