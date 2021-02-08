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

dynamics_cars = []
for i in range(5):
    dynamics_cars.append(partial(_dynamics_car, n=i))

# def dynamics_car_pg(x, u, p):
#     """
#     :param x: state variables
#     :param u: control inputs variables
#     :param p: parameter variables
#     :return:
#     """
#     maxspeed = p[params.p_idx.maxspeed]
#     pspeedcostM = p[params.p_idx.pspeedcostM]
#     pLeftLane = p[params.p_idx.pLeftLane]
#
#     points = getPointsFromParameters(p, params.n_param, params.n_bspline_points)
#     points_c2 = getPointsFromParameters(p, params.n_param, params.n_bspline_points)
#     points_c3 = getPointsFromParameters(p, params.n_param, params.n_bspline_points)
#
#     splx, sply = casadiDynamicBSPLINE(x[params.s_idx.s - params.n_inputs], points)
#     splsx, splsy = casadiDynamicBSPLINEsidewards(x[params.s_idx.s - params.n_inputs], points)
#
#     splx_c2, sply_c2 = casadiDynamicBSPLINE(x[params.s_idx.s_c2 - params.n_inputs], points_c2)
#     splsx_c2, splsy_c2 = casadiDynamicBSPLINEsidewards(x[params.s_idx.s_c2 - params.n_inputs], points_c2)
#
#     splx_c3, sply_c3 = casadiDynamicBSPLINE(x[params.s_idx.s_c3 - params.n_inputs], points_c3)
#     splsx_c3, splsy_c3 = casadiDynamicBSPLINEsidewards(x[params.s_idx.s_c3 - params.n_inputs], points_c3)
#
#     sidewards = vertcat(splsx, splsy)
#     sidewards_c2 = vertcat(splsx_c2, splsy_c2)
#     sidewards_c3 = vertcat(splsx_c3, splsy_c3)
#
#     realPos = vertcat(x[params.s_idx.x - params.n_inputs], x[params.s_idx.y - params.n_inputs])
#     centerPos = realPos
#
#     realPos_c2 = vertcat(x[params.s_idx.x_c2 - params.n_inputs], x[params.s_idx.y_c2 - params.n_inputs])
#     centerPos_c2 = realPos_c2
#
#     realPos_c3 = vertcat(x[params.s_idx.x_c3 - params.n_inputs], x[params.s_idx.y_c3 - params.n_inputs])
#     centerPos_c3 = realPos_c3
#
#     wantedpos = vertcat(splx, sply)
#     error = centerPos - wantedpos
#     laterror = mtimes(sidewards.T, error)
#
#     wantedpos_c2 = vertcat(splx_c2, sply_c2)
#     error_c2 = centerPos_c2 - wantedpos_c2
#     laterror_c2 = mtimes(sidewards_c2.T, error_c2)
#
#     wantedpos_c3 = vertcat(splx_c3, sply_c3)
#     error_c3 = centerPos_c3 - wantedpos_c3
#     laterror_c3 = mtimes(sidewards_c3.T, error_c3)
#
#     speedcostM = speedPunisherA(x[params.s_idx.vx - params.n_inputs], maxspeed) * pspeedcostM
#     leftLaneCost = pLeftLane * laterrorPunisher(laterror, 0)
#
#     speedcostM_c2 = speedPunisherA(x[params.s_idx.vx_c2 - params.n_inputs], maxspeed) * pspeedcostM
#     leftLaneCost_c2 = pLeftLane * laterrorPunisher(laterror_c2, 0)
#
#     speedcostM_c3 = speedPunisherA(x[params.s_idx.vx_c3 - params.n_inputs], maxspeed) * pspeedcostM
#     leftLaneCost_c3 = pLeftLane * laterrorPunisher(laterror_c3, 0)
#
#     dotab = u[params.i_idx.dAb]
#     dotbeta = u[params.i_idx.dBeta]
#     dots = u[params.i_idx.dots]
#     slack = u[params.i_idx.slack]
#
#     dotab_c2 = u[params.i_idx.dAb_c2]
#     dotbeta_c2 = u[params.i_idx.dBeta_c2]
#     dots_c2 = u[params.i_idx.dots_c2]
#     slack_c2 = u[params.i_idx.slack_c2]
#
#     dotab_c3 = u[params.i_idx.dAb_c3]
#     dotbeta_c3 = u[params.i_idx.dBeta_c3]
#     dots_c3 = u[params.i_idx.dots_c3]
#     slack_c3 = u[params.i_idx.slack_c3]
#
#     ab = x[params.s_idx.ab - params.n_inputs]
#     theta = x[params.s_idx.theta - params.n_inputs]
#     vx = x[params.s_idx.vx - params.n_inputs]
#     beta = x[params.s_idx.beta - params.n_inputs]  # from steering.
#
#     ab_c2 = x[params.s_idx.ab_c2 - params.n_inputs]
#     theta_c2 = x[params.s_idx.theta_c2 - params.n_inputs]
#     vx_c2 = x[params.s_idx.vx_c2 - params.n_inputs]
#     beta_c2 = x[params.s_idx.beta_c2 - params.n_inputs]  # from steering.
#
#     ab_c3 = x[params.s_idx.ab_c3 - params.n_inputs]
#     theta_c3 = x[params.s_idx.theta_c3 - params.n_inputs]
#     vx_c3 = x[params.s_idx.vx_c3 - params.n_inputs]
#     beta_c3 = x[params.s_idx.beta_c3 - params.n_inputs]  # from steering.
#
#     if isinstance(x[0], float):
#         dx = np.zeros(params.n_states)
#     else:  # fixme not sure what this was for
#         dx = SX.zeros(params.n_states, 1)
#     lc = p[params.p_idx.carLength]
#     pslack = p[params.p_idx.pslack]
#
#     dx[params.s_idx.x - params.n_inputs] = vx * cos(theta)
#     dx[params.s_idx.y - params.n_inputs] = vx * sin(theta)
#     dx[params.s_idx.theta - params.n_inputs] = vx * tan(beta) / lc
#     dx[params.s_idx.vx - params.n_inputs] = ab
#     dx[params.s_idx.ab - params.n_inputs] = dotab
#     dx[params.s_idx.beta - params.n_inputs] = dotbeta
#     dx[params.s_idx.s - params.n_inputs] = dots
#     dx[params.s_idx.CumSlackCost - params.n_inputs] = pslack * slack
#     dx[params.s_idx.CumLatSpeedCost - params.n_inputs] = speedcostM + leftLaneCost
#
#     dx[params.s_idx.x_c2 - params.n_inputs] = vx_c2 * cos(theta_c2)
#     dx[params.s_idx.y_c2 - params.n_inputs] = vx_c2 * sin(theta_c2)
#     dx[params.s_idx.theta_c2 - params.n_inputs] = vx_c2 * tan(beta_c2) / lc
#     dx[params.s_idx.vx_c2 - params.n_inputs] = ab_c2
#     dx[params.s_idx.ab_c2 - params.n_inputs] = dotab_c2
#     dx[params.s_idx.beta_c2 - params.n_inputs] = dotbeta_c2
#     dx[params.s_idx.s_c2 - params.n_inputs] = dots_c2
#     dx[params.s_idx.CumSlackCost_c2 - params.n_inputs] = pslack * slack_c2
#     dx[params.s_idx.CumLatSpeedCost_c2 - params.n_inputs] = speedcostM_c2 + leftLaneCost_c2
#
#     dx[params.s_idx.x_c3 - params.n_inputs] = vx_c3 * cos(theta_c3)
#     dx[params.s_idx.y_c3 - params.n_inputs] = vx_c3 * sin(theta_c3)
#     dx[params.s_idx.theta_c3 - params.n_inputs] = vx_c3 * tan(beta_c3) / lc
#     dx[params.s_idx.vx_c3 - params.n_inputs] = ab_c3
#     dx[params.s_idx.ab_c3 - params.n_inputs] = dotab_c3
#     dx[params.s_idx.beta_c3 - params.n_inputs] = dotbeta_c3
#     dx[params.s_idx.s_c3 - params.n_inputs] = dots_c3
#     dx[params.s_idx.CumSlackCost_c3 - params.n_inputs] = pslack * slack_c3
#     dx[params.s_idx.CumLatSpeedCost_c3 - params.n_inputs] = speedcostM_c3 + leftLaneCost_c3
#     return dx
