from dynamics.dynamics import kitt_dynamics
from dynamics.dynamics_utils import ackermann_map, rotmat
from bspline.bspline import *
from .car_util import *
from Car_mpcc.Car_Model import params
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

    splx, sply = casadiDynamicBSPLINE(x[params.s_idx.s - params.n_inputs], points)
    splsx, splsy = casadiDynamicBSPLINEsidewards(x[params.s_idx.s - params.n_inputs], points)

    sidewards = vertcat(splsx, splsy)

    realPos = vertcat(x[params.s_idx.x - params.n_inputs], x[params.s_idx.y - params.n_inputs])
    centerPos = realPos
    wantedpos = vertcat(splx, sply)
    error = centerPos - wantedpos
    laterror = mtimes(sidewards.T, error)

    pLeftLane = p[params.p_idx.pLeftLane]
    pSpeedCostM = p[params.p_idx.pSpeedCostM]
    maxSpeed = p[params.p_idx.maxSpeed]
    lc = p[params.p_idx.carLength]
    pSlack = p[params.p_idx.pSlack]

    dotab = u[params.i_idx.dAb]
    ab = x[params.s_idx.ab - params.n_inputs]
    dotbeta = u[params.i_idx.dBeta]
    ds = u[params.i_idx.ds]

    theta = x[params.s_idx.theta - params.n_inputs]
    v = x[params.s_idx.v - params.n_inputs]
    beta = x[params.s_idx.beta - params.n_inputs]  # from steering.
    slack = u[params.i_idx.slack]

    leftLaneCost = pLeftLane * latErrorPunisher(laterror)
    speedCost = pSpeedCostM * speedPunisher(v, maxSpeed)

    collisionCost = pSlack * slack


    if isinstance(x[0], float):
        dx = np.zeros(params.n_states)
    else:  # fixme not sure what this was for
        dx = SX.zeros(params.n_states, 1)

    dx[params.s_idx.x - params.n_inputs] = v * cos(theta)
    dx[params.s_idx.y - params.n_inputs] = v * sin(theta)
    dx[params.s_idx.theta - params.n_inputs] = v * tan(beta) / lc
    dx[params.s_idx.v - params.n_inputs] = ab
    dx[params.s_idx.ab - params.n_inputs] = dotab
    dx[params.s_idx.beta - params.n_inputs] = dotbeta
    dx[params.s_idx.s - params.n_inputs] = ds
    dx[params.s_idx.latSpeedCost-params.n_inputs] = leftLaneCost + speedCost
    dx[params.s_idx.collCost - params.n_inputs] = collisionCost
    return dx
