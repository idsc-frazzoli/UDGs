from dataclasses import dataclass
from math import floor

from bspline.bspline import casadiDynamicBSPLINE, atan2, casadiDynamicBSPLINEforward
from Car_mpcc.forces_utils import ForcesException
from Car_mpcc.Car_Model import params
import numpy as np

from Car_mpcc.Car_Model.driver_config import behaviors_zoo
from Car_mpcc.Car_Model.car_util import set_p_car, casadiGetMaxAcc
from tracks import Track
from tracks.utils import spline_progress_from_pose

from tracks.zoo import winti_001
from vehicle import gokart_pool, KITT


@dataclass(frozen=True)
class SimData:
    x: np.ndarray
    x_pred: np.ndarray
    u: np.ndarray
    u_pred: np.ndarray
    next_spline_points: np.ndarray
    track: Track
    solver_it: np.ndarray
    solver_time: np.ndarray


def sim_car_model(
        model, solver, sim_length: int = 200, seed: int = 1, track: Track = winti_001
) -> SimData:
    """

    :param seed:
    :param track:
    :param model:
    :param solver:
    :param sim_length:
    :return:
    """
    # Load some parameters
    # front_pacejka = gokart_pool[KITT].front_tires.pacejka
    # rear_pacejka = gokart_pool[KITT].rear_tires.pacejka

    # steering_column = gokart_pool[KITT].steering
    behavior = behaviors_zoo["beginner"].config
    n_states = params.n_states
    n_inputs = params.n_inputs
    x_idx = params.s_idx

    # Variables for storing simulation data
    x = np.zeros((n_states, sim_length + 1))  # states
    u = np.zeros((n_inputs, sim_length))  # inputs
    x_pred = np.zeros((n_states, model.N, sim_length + 1))  # states
    u_pred = np.zeros((n_inputs, model.N, sim_length))  # inputs
    next_spline_points = np.zeros((params.n_bspline_points, 3, sim_length))
    solver_it = np.zeros(sim_length)
    solver_time = np.zeros(sim_length)

    # Set initial condition
    init_from_progress = False
    spline_start_idx = 0
    if init_from_progress:
        init_progress = 0.0
        x_pos, y_pos = casadiDynamicBSPLINE(init_progress, track.spline.as_np_array())
        dx, dy = casadiDynamicBSPLINEforward(init_progress, track.spline.as_np_array())
        theta_pos = atan2(dy, dx)
    else:
        # init progress from pose
        x_pos, y_pos = 32.68, 19.10
        theta_pos = 0.039
        init_progress = spline_progress_from_pose(spline_track=track.spline, pose=np.array([x_pos, y_pos, theta_pos]))

    xinit = np.zeros(n_states)
    xinit[x_idx.x - n_inputs] = x_pos
    xinit[x_idx.y - n_inputs] = y_pos
    xinit[x_idx.theta - n_inputs] = theta_pos
    xinit[x_idx.v - n_inputs] = 8  # totally arbitrary
    # xinit[x_idx.vy - n_inputs] = -0.001  # totally arbitrary
    xinit[x_idx.beta - n_inputs] = -0.67  # totally arbitrary
    xinit[x_idx.s - n_inputs] = init_progress
    xinit[x_idx.latSpeedCost - n_inputs] = 0
    xinit[x_idx.collCost - n_inputs] = 0
    # xinit[x_idx.ab - n_inputs] = 0

    x[:, 0] = xinit
    problem = {}
    problem["x0"] = np.zeros(model.N * model.nvar)

    for k in range(sim_length):

        # find bspline
        # while x[x_idx.s - n_inputs, k] >= 1:
        while x[x_idx.s - n_inputs, k] >= 1:
            # spline step forward
            spline_start_idx += 1  # fixme some modulo operation (number of control points) is probably needed
            x[x_idx.s - n_inputs, k] -= 1

        for i in range(params.n_bspline_points):
            next_point = track.spline.get_control_point(spline_start_idx + i)
            next_spline_points[i, :, k] = next_point

        # Limit acceleration
        x[x_idx.ab - n_inputs, k] = min(
            casadiGetMaxAcc(x[x_idx.v - n_inputs, k]) - 0.0001, x[x_idx.ab - n_inputs, k]
        )  # fixme why -0.0001?

        # Set initial state
        problem["xinit"] = x[:, k]
        # Set runtime parameters (the only really changing between stages are the next control points of the spline)
        p_vector = set_p_car(
            targetSpeed=behavior.targetSpeed,
            maxSpeed=behavior.maxSpeed,
            firstOptCost=behavior.firstOptCost,
            secondOptCost=behavior.secondOptCost,
            progressMax=behavior.progressMax,
            obstaclePosX=behavior.obstaclePosX,
            obstaclePosY=behavior.obstaclePosY,
            pLag=behavior.pLag,
            pLat=behavior.pLat,
            pLeftLane=behavior.pLeftLane,
            pSpeedCostB=behavior.pSpeedCostB,
            pSpeedCostA=behavior.pSpeedCostA,
            pSpeedCostM=behavior.pSpeedCostM,
            pab=behavior.pab,
            pDotBeta=behavior.pDotBeta,
            pSlack=behavior.pSlack,
            distance=behavior.distance,
            carLength=behavior.carLength,
            points=next_spline_points[:, :, k],
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
        u_pred[:, :, k] = temp[0:n_inputs, :]  # predicted inputs
        x_pred[:, :, k] = temp[n_inputs: params.n_var, :]  # predicted states

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
        solver_it=solver_it,
        solver_time=solver_time,
    )
