from functools import partial

from .car_util import *
from bspline.bspline import *
from Car_mpcc.Car_optimizer import params
from casadi import *

import numpy as np


def _dynamics_car(x, u, p, n):
    """todo

    :param x: state variables
    :param u: control inputs variables
    :param p: parameter variables
    :return:
    """
    maxspeed = p[params.p_idx.maxspeed]
    pspeedcostM = p[params.p_idx.pspeedcostM]
    pLeftLane = p[params.p_idx.pLeftLane]
    lc = p[params.p_idx.carLength]
    pslack = p[params.p_idx.pslack]

    speedcostM = np.zeros(n)
    leftLaneCost = np.zeros(n)

    dotab = np.zeros(n)
    dotbeta = np.zeros(n)
    dots = np.zeros(n)
    slack = np.zeros(n)
    ab = np.zeros(n)
    theta = np.zeros(n)
    vx = np.zeros(n)
    beta = np.zeros(n)

    pointsO = params.n_param
    pointsN = params.n_bspline_points

    # points = np.zeros((n, pointsN, 2))

    if isinstance(x[0], float):
        dx = np.zeros(n * params.n_states)
    else:  # fixme not sure what this was for
        dx = SX.zeros(n * params.n_states, 1)

    for k in range(n):
        upd_s_idx = k * params.n_states - n * params.n_inputs
        upd_i_idx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN, pointsN)
        splx, sply = casadiDynamicBSPLINE(x[params.s_idx.s + upd_s_idx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(x[params.s_idx.s + upd_s_idx], points)

        sidewards = vertcat(splsx, splsy)
        realPos = vertcat(x[params.s_idx.x + upd_s_idx], x[params.s_idx.y + upd_s_idx])
        centerPos = realPos
        wantedpos = vertcat(splx, sply)
        error = centerPos - wantedpos
        laterror = mtimes(sidewards.T, error)
        speedcostM[k] = speedPunisherA(x[params.s_idx.vx + upd_s_idx], maxspeed) * pspeedcostM
        leftLaneCost[k] = pLeftLane * laterrorPunisher(laterror, 0)

        dotab[k] = u[params.i_idx.dAb + upd_i_idx]
        dotbeta[k] = u[params.i_idx.dBeta + upd_i_idx]
        dots[k] = u[params.i_idx.dots + upd_i_idx]
        slack[k] = u[params.i_idx.slack + upd_i_idx]
        ab[k] = x[params.s_idx.ab + upd_s_idx]
        theta[k] = x[params.s_idx.theta + upd_s_idx]
        vx[k] = x[params.s_idx.vx + upd_s_idx]
        beta[k] = x[params.s_idx.beta + upd_s_idx]  # from steering.

        dx[params.s_idx.x + upd_s_idx] = vx[k] * cos(theta[k])
        dx[params.s_idx.y + upd_s_idx] = vx[k] * sin(theta[k])
        dx[params.s_idx.theta + upd_s_idx] = vx[k] * tan(beta[k]) / lc
        dx[params.s_idx.vx + upd_s_idx] = ab[k]
        dx[params.s_idx.ab + upd_s_idx] = dotab[k]
        dx[params.s_idx.beta + upd_s_idx] = dotbeta[k]
        dx[params.s_idx.s + upd_s_idx] = dots[k]
        dx[params.s_idx.CumSlackCost + upd_s_idx] = pslack * slack[k]
        dx[params.s_idx.CumLatSpeedCost + upd_s_idx] = speedcostM[k] + leftLaneCost[k]

    return dx


dynamics_cars = []
for i in range(5):
    dynamics_cars.append(partial(_dynamics_car, n=i))
