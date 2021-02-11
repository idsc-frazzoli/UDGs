from dataclasses import dataclass
from math import floor

from bspline.bspline import casadiDynamicBSPLINE, atan2, casadiDynamicBSPLINEforward
from Car_mpcc.forces_utils import ForcesException
from Car_mpcc.Car_optimizer import params
import numpy as np

from Car_mpcc.Car_optimizer.driver_config import behaviors_zoo
from Car_mpcc.Car_optimizer.car_util import set_p_car, casadiGetMaxAcc
from tracks import Track
from tracks.utils import spline_progress_from_pose

from tracks.zoo import winti_001, straightLineR2L, straightLineN2S
from vehicle import gokart_pool, KITT


@dataclass(frozen=True)
class SimData:
    x: np.ndarray
    x_pred: np.ndarray
    u: np.ndarray
    u_pred: np.ndarray
    next_spline_points: np.ndarray
    track: Track
    track2: Track
    track3: Track
    track4: Track
    track5: Track
    solver_it: np.ndarray
    solver_time: np.ndarray


def sim_car_model(
        model, solver, num_cars, sim_length: int = 200, seed: int = 1, track: Track = straightLineR2L,
        track2: Track = straightLineN2S, track3: Track = straightLineR2L, track4: Track = straightLineR2L,
        track5: Track = straightLineR2L) -> SimData:
    """

    :param seed:
    :param track:
    :param track2:
    :param track3:
    :param track4:
    :param track5:
    :param model:
    :param solver:
    :param sim_length:
    :return:
    """
    # Load some parameters

    behavior = behaviors_zoo["Config1"].config
    n_states = params.n_states
    n_inputs = params.n_inputs
    x_idx = params.s_idx

    # Variables for storing simulation data
    x = np.zeros((n_states * num_cars, sim_length + 1))  # states
    u = np.zeros((n_inputs * num_cars, sim_length))  # inputs
    x_pred = np.zeros((n_states * num_cars, model.N, sim_length + 1))  # states
    u_pred = np.zeros((n_inputs * num_cars, model.N, sim_length))  # inputs
    next_spline_points = np.zeros((params.n_bspline_points * num_cars, 3, sim_length))
    solver_it = np.zeros(sim_length)
    solver_time = np.zeros(sim_length)

    # Set initial condition
    init_from_progress = True
    spline_start_idx = 0
    if init_from_progress:
        init_progress = 0.0
        x_pos = np.zeros(num_cars)
        y_pos = np.zeros(num_cars)
        dx = np.zeros(num_cars)
        dy = np.zeros(num_cars)
        theta_pos = np.zeros(num_cars)
        for k in range(num_cars):
            if k == 0:
                x_pos[k], y_pos[k] = casadiDynamicBSPLINE(init_progress, track.spline.as_np_array())
                dx[k], dy[k] = casadiDynamicBSPLINEforward(init_progress, track.spline.as_np_array())
                theta_pos[k] = atan2(dy[k], dx[k])
            elif k == 1:
                x_pos[k], y_pos[k] = casadiDynamicBSPLINE(init_progress, track2.spline.as_np_array())
                dx[k], dy[k] = casadiDynamicBSPLINEforward(init_progress, track2.spline.as_np_array())
                theta_pos[k] = atan2(dy[k], dx[k])
            elif k == 2:
                x_pos[k], y_pos[k] = casadiDynamicBSPLINE(init_progress, track3.spline.as_np_array())
                dx[k], dy[k] = casadiDynamicBSPLINEforward(init_progress, track3.spline.as_np_array())
                theta_pos[k] = atan2(dy[k], dx[k])
            elif k == 3:
                x_pos[k], y_pos[k] = casadiDynamicBSPLINE(init_progress, track4.spline.as_np_array())
                dx[k], dy[k] = casadiDynamicBSPLINEforward(init_progress, track4.spline.as_np_array())
                theta_pos[k] = atan2(dy[k], dx[k])
            elif k == 4:
                x_pos[k], y_pos[k] = casadiDynamicBSPLINE(init_progress, track5.spline.as_np_array())
                dx[k], dy[k] = casadiDynamicBSPLINEforward(init_progress, track5.spline.as_np_array())
                theta_pos[k] = atan2(dy[k], dx[k])
    else:
        # init progress from pose
        x_pos, y_pos = 32.68, 19.10
        theta_pos = 0.039
        init_progress = spline_progress_from_pose(spline_track=track.spline, pose=np.array([x_pos, y_pos, theta_pos]))

    xinit = np.zeros(n_states * num_cars)
    for k in range(num_cars):
        upd_s_idx = - n_inputs + k * n_states
        xinit[x_idx.x + upd_s_idx] = x_pos[k]
        xinit[x_idx.y + upd_s_idx] = y_pos[k]
        xinit[x_idx.theta + upd_s_idx] = theta_pos[k]
        xinit[x_idx.vx + upd_s_idx] = 0.0  # totally arbitrary
        xinit[x_idx.beta + upd_s_idx] = -0.67  # totally arbitrary
        xinit[x_idx.s + upd_s_idx] = init_progress

    x[:, 0] = xinit
    problem = {}
    problem["x0"] = np.zeros(model.N * model.nvar)
    spline_start_idx = np.zeros(num_cars)
    for k in range(sim_length):

        # find bspline
        # while x[x_idx.s - n_inputs, k] >= 1:
        for jj in range(num_cars):
            upd_s_idx = - n_inputs + jj * n_states
            while x[x_idx.s + upd_s_idx, k] >= 1:
                # spline step forward
                spline_start_idx[jj] += 1  # fixme some module operation (number of control points) is probably needed
                x[x_idx.s + upd_s_idx, k] -= 1
            if jj == 0:
                for i in range(params.n_bspline_points):
                    next_point = track.spline.get_control_point(spline_start_idx[jj].astype(int) + i)
                    next_spline_points[i + jj * params.n_bspline_points, :, k] = next_point
            elif jj == 1:
                for i in range(params.n_bspline_points):
                    next_point = track2.spline.get_control_point(spline_start_idx[jj].astype(int) + i)
                    next_spline_points[i + jj * params.n_bspline_points, :, k] = next_point
            elif jj == 2:
                for i in range(params.n_bspline_points):
                    next_point = track3.spline.get_control_point(spline_start_idx[jj].astype(int) + i)
                    next_spline_points[i + jj * params.n_bspline_points, :, k] = next_point
            elif jj == 3:
                for i in range(params.n_bspline_points):
                    next_point = track4.spline.get_control_point(spline_start_idx[jj].astype(int) + i)
                    next_spline_points[i + jj * params.n_bspline_points, :, k] = next_point
            else:
                for i in range(params.n_bspline_points):
                    next_point = track5.spline.get_control_point(spline_start_idx[jj].astype(int) + i)
                    next_spline_points[i + jj * params.n_bspline_points, :, k] = next_point
            # Limit acceleration
            x[x_idx.ab + upd_s_idx, k] = min(
                casadiGetMaxAcc(x[x_idx.vx + upd_s_idx, k]) - 0.0001, x[x_idx.ab + upd_s_idx, k]
            )  # fixme why -0.0001?

        # Set initial state
        problem["xinit"] = x[:, k]
        # Set runtime parameters (the only really changing between stages are the next control points of the spline)
        p_vector = set_p_car(
            maxspeed=behavior.maxspeed,
            targetspeed=behavior.targetspeed,
            optcost1=behavior.optcost1,
            optcost2=behavior.optcost2,
            Xobstacle=behavior.Xobstacle,
            Yobstacle=behavior.Yobstacle,
            targetprog=behavior.targetprog,
            pspeedcostA=behavior.pspeedcostA,
            pspeedcostB=behavior.pspeedcostB,
            pspeedcostM=behavior.pspeedcostM,
            plag=behavior.plag,
            plat=behavior.plat,
            pLeftLane=behavior.pLeftLane,
            pab=behavior.pab,
            pdotbeta=behavior.pdotbeta,
            carLength=behavior.carLength,
            distance=behavior.distance,
            pslack=behavior.pslack,
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
            solver_it[k] = info.it
            solver_time[k] = info.solvetime

        # Extract output and initialize next iteration with current solution shifted by one stage
        problem["x0"][0: model.nvar * (model.N - 1)] = output["all_var"][model.nvar:model.nvar * model.N]
        problem["x0"][model.nvar * (model.N - 1): model.nvar * model.N] = output["all_var"][model.nvar * (
                model.N - 1):model.nvar * model.N]
        temp = output["all_var"].reshape(model.nvar, model.N, order='F')
        u_pred[:, :, k] = temp[0:n_inputs * num_cars, :]  # predicted inputs
        x_pred[:, :, k] = temp[n_inputs * num_cars: params.n_var * num_cars, :]  # predicted states

        # Apply optimized input u of first stage to system and save simulation data
        u[:, k] = u_pred[:, 0, k]
        z = np.concatenate((u[:, k], x[:, k]))

        # "simulate" state evolution
        x[:, k + 1] = np.transpose(model.eq(z, p_vector))

    return SimData(
        x=x,
        x_pred=x_pred,
        u=u,
        u_pred=u_pred,
        next_spline_points=next_spline_points,
        track=track,
        track2=track2,
        track3=track3,
        track4=track4,
        track5=track5,
        solver_it=solver_it,
        solver_time=solver_time,
    )
