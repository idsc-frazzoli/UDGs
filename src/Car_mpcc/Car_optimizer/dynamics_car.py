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
    points = np.zeros((n, params.n_bspline_points, 2))

    for k in range(n):
        points[k, :, :] = getPointsFromParameters(p, params.n_param, params.n_bspline_points)
        splx, sply = casadiDynamicBSPLINE(x[params.s_idx.s + k*params.n_states - n*params.n_inputs], points[k])
        splsx, splsy = casadiDynamicBSPLINEsidewards(x[params.s_idx.s + k*params.n_states - n*params.n_inputs], points[k])

        sidewards = vertcat(splsx, splsy)
        realPos = vertcat(x[params.s_idx.x + k*params.n_states - n*params.n_inputs], x[params.s_idx.y + k*params.n_states - n*params.n_inputs])
        centerPos = realPos
        wantedpos = vertcat(splx, sply)
        error = centerPos - wantedpos
        laterror = mtimes(sidewards.T, error)
        speedcostM[k] = speedPunisherA(x[params.s_idx.vx + k*params.n_states - n*params.n_inputs], maxspeed) * pspeedcostM
        leftLaneCost[k] = pLeftLane * laterrorPunisher(laterror, 0)

    dotab = np.zeros(n)
    dotbeta = np.zeros(n)
    dots = np.zeros(n)
    slack = np.zeros(n)
    ab = np.zeros(n)
    theta = np.zeros(n)
    vx = np.zeros(n)
    beta = np.zeros(n)

    if isinstance(x[0], float):
        dx = np.zeros(n * params.n_states)
    else:  # fixme not sure what this was for
        dx = SX.zeros(n * params.n_states, 1)

    for k in range(n):
        dotab[k] = u[params.i_idx.dAb + k*params.n_inputs]
        dotbeta[k] = u[params.i_idx.dBeta + k*params.n_inputs]
        dots[k] = u[params.i_idx.dots + k*params.n_inputs]
        slack[k] = u[params.i_idx.slack + k*params.n_inputs]
        ab[k] = x[params.s_idx.ab + k*params.n_states - n*params.n_inputs]
        theta[k] = x[params.s_idx.theta + k*params.n_states - n*params.n_inputs]
        vx[k] = x[params.s_idx.vx + k*params.n_states - n*params.n_inputs]
        beta[k] = x[params.s_idx.beta + k*params.n_states - n*params.n_inputs]  # from steering.

        dx[params.s_idx.x + k*params.n_states - n*params.n_inputs] = vx[k] * cos(theta[k])
        dx[params.s_idx.y + k*params.n_states - n*params.n_inputs] = vx[k] * sin(theta[k])
        dx[params.s_idx.theta + k*params.n_states - n*params.n_inputs] = vx[k] * tan(beta[k]) / lc
        dx[params.s_idx.vx + k*params.n_states - n*params.n_inputs] = ab[k]
        dx[params.s_idx.ab + k*params.n_states - n*params.n_inputs] = dotab[k]
        dx[params.s_idx.beta + k*params.n_states - n*params.n_inputs] = dotbeta[k]
        dx[params.s_idx.s + k*params.n_states - n*params.n_inputs] = dots[k]
        dx[params.s_idx.CumSlackCost + k*params.n_states - n*params.n_inputs] = pslack * slack[k]
        dx[params.s_idx.CumLatSpeedCost + k*params.n_states - n*params.n_inputs] = speedcostM[k] + leftLaneCost[k]

    return dx

dynamics_cars = []
for i in range(7):
    dynamics_cars.append(partial(_dynamics_car, n=i))
