from .car_util import *
from bspline.bspline import *
from Car_mpcc.Car_optimizer import params
from casadi import *

import numpy as np


def dynamics_car(x, u, p):
    """todo



    :param x: state variables
    :param u: control inputs variables
    :param p: parameter variables
    :return:
    """
    points = getPointsFromParameters(p, params.n_param, params.n_bspline_points)

    maxspeed = p[params.p_idx.maxspeed]
    pspeedcostM = p[params.p_idx.pspeedcostM]

    splx, sply = casadiDynamicBSPLINE(x[params.s_idx.s - params.n_inputs], points)
    splsx, splsy = casadiDynamicBSPLINEsidewards(x[params.s_idx.s - params.n_inputs], points)

    sidewards = vertcat(splsx, splsy)

    realPos = vertcat(x[params.s_idx.x - params.n_inputs], x[params.s_idx.y - params.n_inputs])
    centerPos = realPos
    pLeftLane = p[params.p_idx.pLeftLane]

    wantedpos = vertcat(splx, sply)
    error = centerPos - wantedpos
    laterror = mtimes(sidewards.T, error)
    speedcostM = speedPunisherA(x[params.s_idx.vx - params.n_inputs], maxspeed) * pspeedcostM

    leftLaneCost = pLeftLane * laterrorPunisher(laterror, 0)

    dotab = u[params.i_idx.dAb]
    dotbeta = u[params.i_idx.dBeta]
    dots = u[params.i_idx.dots]
    slack = u[params.i_idx.slack]

    ab = x[params.s_idx.ab - params.n_inputs]
    theta = x[params.s_idx.theta - params.n_inputs]
    vx = x[params.s_idx.vx - params.n_inputs]
    beta = x[params.s_idx.beta - params.n_inputs]  # from steering.

    if isinstance(x[0], float):
        dx = np.zeros(params.n_states)
    else:  # fixme not sure what this was for
        dx = SX.zeros(params.n_states, 1)
    lc = p[params.p_idx.carLength]
    pslack = p[params.p_idx.pslack]

    dx[params.s_idx.x - params.n_inputs] = vx * cos(theta)
    dx[params.s_idx.y - params.n_inputs] = vx * sin(theta)
    dx[params.s_idx.theta - params.n_inputs] = vx * tan(beta) / lc
    dx[params.s_idx.vx - params.n_inputs] = ab
    dx[params.s_idx.ab - params.n_inputs] = dotab
    dx[params.s_idx.beta - params.n_inputs] = dotbeta
    dx[params.s_idx.s - params.n_inputs] = dots
    dx[params.s_idx.CumSlackCost - params.n_inputs] = pslack * slack
    dx[params.s_idx.CumLatSpeedCost - params.n_inputs] = speedcostM + leftLaneCost
    return dx
