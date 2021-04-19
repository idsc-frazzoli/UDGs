from functools import partial

from .car_util import *
from udgs.bspline.bspline import *
from udgs.forces_models.model_def import params
from casadi import *

import numpy as np


def _dynamics_car(x, u, p, n):
    """todo

    :param x: state variables X, Y, theta, Vx, Acc_x, Delta, S, CumSlackCost, CumLatSpeedCost
    :param u: control inputs variables dAcc, dDelta, dS, slack, slack_coll, slack_obs
    :param p: parameter variables
    :param n: number of vehicles
    :return:
    """
    speed_limit = p[params.p_idx.SpeedLimit]
    kAboveSpeedLimit = p[params.p_idx.kAboveSpeedLimit]
    pLeftLane = p[params.p_idx.pLeftLane]
    lc = p[params.p_idx.carLength]
    kSlack = p[params.p_idx.kSlack]
    kLag = p[params.p_idx.kLag]
    pointsO = params.n_opt_param
    pointsN = params.n_bspline_points

    # points = np.zeros((n, pointsN, 2))

    if isinstance(x[0], float):
        dx = np.zeros(n * params.n_states)
    else:
        dx = SX.zeros(n * params.n_states, 1)

    for k in range(n):
        upd_s_idx = k * params.n_states - params.n_inputs
        upd_i_idx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN * 3, pointsN)
        splx, sply = casadiDynamicBSPLINE(x[params.x_idx.S + upd_s_idx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(x[params.x_idx.S + upd_s_idx], points)
        spldx, spldy = casadiDynamicBSPLINEforward(x[params.x_idx.S + upd_s_idx], points)

        forward = vertcat(spldx, spldy)
        sidewards = vertcat(splsx, splsy)
        realPos = vertcat(x[params.x_idx.X + upd_s_idx], x[params.x_idx.Y + upd_s_idx])
        centerPos = realPos
        wantedpos = vertcat(splx, sply)
        error = centerPos - wantedpos
        laterror = mtimes(sidewards.T, error)
        speedcostM = speedPunisherA(x[params.x_idx.Vx + upd_s_idx], speed_limit) * kAboveSpeedLimit
        leftLaneCost = pLeftLane * laterrorPunisher(laterror, 0)
        lagerror = mtimes(forward.T, error)
        lagcost = kLag * lagerror ** 2

        dAcc = u[params.u_idx.dAcc + upd_i_idx]
        dDelta = u[params.u_idx.dDelta + upd_i_idx]
        dS = u[params.u_idx.dS + upd_i_idx]
        slack = u[params.u_idx.Slack_Lat + upd_i_idx]
        slack_coll = u[params.u_idx.Slack_Coll + upd_i_idx]
        slack_obs = u[params.u_idx.Slack_Obs + upd_i_idx]

        theta = x[params.x_idx.Theta + upd_s_idx]
        vx = x[params.x_idx.Vx + upd_s_idx]
        acc = x[params.x_idx.Acc + upd_s_idx]
        delta = x[params.x_idx.Delta + upd_s_idx]  # from steering.

        dx[params.x_idx.X + upd_s_idx] = vx * cos(theta)
        dx[params.x_idx.Y + upd_s_idx] = vx * sin(theta)
        dx[params.x_idx.Theta + upd_s_idx] = vx * tan(delta) / lc
        dx[params.x_idx.Vx + upd_s_idx] = acc
        dx[params.x_idx.Acc + upd_s_idx] = dAcc
        dx[params.x_idx.Delta + upd_s_idx] = dDelta
        dx[params.x_idx.S + upd_s_idx] = dS
        dx[params.x_idx.CumSlackCost + upd_s_idx] = kSlack * slack + kSlack * slack_coll + kSlack * slack_obs
        dx[params.x_idx.CumLatSpeedCost + upd_s_idx] = speedcostM + leftLaneCost

    return dx


dynamics_cars = []
for i in range(5):
    dynamics_cars.append(partial(_dynamics_car, n=i))
