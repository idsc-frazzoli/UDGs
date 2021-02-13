from functools import partial

from .car_util import *
from bspline.bspline import *
from casadi import *
import numpy as np


def _nlconst_car(z, p, n):
    """
    Computes the nonlinear inequality constraints.

    :param z: States and inputs of the system dynamics
    :param p: Parameters of the system dynamics
    :return: List containing the constraints #todo
    """

    pointsO = params.n_param
    pointsN = params.n_bspline_points
    v = []
    coll_constraints = int(n*(n-1)/2)
    minSafetyDistance = p[params.p_idx.minSafetyDistance]

    for k in range(n):
        upd_s_idx = k * params.n_states + (n - 1) * params.n_inputs
        upd_i_idx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN * 3, pointsN)
        radii = getRadiiFromParameters(p, pointsO + k * pointsN * 3, pointsN)

        splx, sply = casadiDynamicBSPLINE(z[params.s_idx.S + upd_s_idx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.S + upd_s_idx], points)
        r = casadiDynamicBSPLINERadius(z[params.s_idx.S + upd_s_idx], radii)

        wantedpos = vertcat(splx, sply)
        sidewards = vertcat(splsx, splsy)

        realPos = np.array([z[params.s_idx.X + upd_s_idx], z[params.s_idx.Y + upd_s_idx]])
        centerPos = realPos
        error = centerPos - wantedpos
        laterror = mtimes(sidewards.T, error)

        slack = z[params.i_idx.Slack_Lat + upd_i_idx]

        v1 = laterror - r - slack
        v2 = -laterror - r - slack

        v = np.append(v, np.array([v1, v2]))

    if n == 2:
        upd_s_idx1 = params.n_inputs
        upd_s_idx2 = params.n_states + params.n_inputs
        #  upd_i_idx = params.n_inputs
        distance_x = (z[params.s_idx.X + upd_s_idx1] - z[params.s_idx.X + upd_s_idx2])  #
        distance_y = (z[params.s_idx.Y + upd_s_idx1] - z[params.s_idx.Y + upd_s_idx2])  #
        eucl_dist = sqrt(distance_x ** 2 + distance_y ** 2)
        slack_coll = z[params.i_idx.Slack_Coll]
        v4 = -eucl_dist + minSafetyDistance - slack_coll
        v = np.append(v, np.array([v4]))
    elif n > 2:
        for kk in range(coll_constraints - 1):
            for jj in range(kk + 1, coll_constraints):
                upd_s_idx1 = kk * params.n_states + (n - 1) * params.n_inputs
                upd_s_idx2 = jj * params.n_states + (n - 1) * params.n_inputs
                upd_i_idx = kk * params.n_inputs
                distance_x = (z[params.s_idx.X + upd_s_idx1] - z[params.s_idx.X + upd_s_idx2])
                distance_y = (z[params.s_idx.Y + upd_s_idx1] - z[params.s_idx.Y + upd_s_idx2])
                slack_coll = z[params.i_idx.Slack_Coll + upd_i_idx]
                v4 = -sqrt(distance_x ** 2 + distance_y ** 2) + minSafetyDistance - slack_coll
                v = np.append(v, np.array([v4]))
    return v


nlconst_car = []
for i in range(5):
    nlconst_car.append(partial(_nlconst_car, n=i))


def _nlconst_carN(z, p, n):
    """
    Computes the nonlinear inequality constraints.

    :param z: States and inputs of the system dynamics
    :param p: Parameters of the system dynamics
    :return: List containing the constraints #todo
    """
    coll_constraints = int(n * (n - 1) / 2)
    minSafetyDistance = p[params.p_idx.minSafetyDistance]
    pointsO = params.n_param
    pointsN = params.n_bspline_points
    v = []

    for k in range(n):
        upd_s_idx = k * params.n_states + (n - 1) * params.n_inputs
        upd_i_idx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN * 3, pointsN)
        radii = getRadiiFromParameters(p, pointsO + k * pointsN * 3, pointsN)

        splx, sply = casadiDynamicBSPLINE(z[params.s_idx.S + upd_s_idx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.S + upd_s_idx], points)
        r = casadiDynamicBSPLINERadius(z[params.s_idx.S + upd_s_idx], radii)

        wantedpos = vertcat(splx, sply)
        sidewards = vertcat(splsx, splsy)

        realPos = np.array([z[params.s_idx.X + upd_s_idx], z[params.s_idx.Y + upd_s_idx]])
        centerPos = realPos
        error = centerPos - wantedpos
        laterror = mtimes(sidewards.T, error)

        slack = z[params.i_idx.Slack_Lat + upd_i_idx]

        v1 = laterror - r - slack
        v2 = -laterror - r - slack
        v3 = -z[params.s_idx.S + upd_s_idx] + p[params.p_idx.TargetProg]

        v = np.append(v, np.array([v1, v2, v3]))

    if n == 2:
        upd_s_idx1 = params.n_inputs
        upd_s_idx2 = params.n_states + params.n_inputs
        # upd_i_idx = params.n_inputs
        distance_x = (z[params.s_idx.X + upd_s_idx1] - z[params.s_idx.X + upd_s_idx2])  #
        distance_y = (z[params.s_idx.Y + upd_s_idx1] - z[params.s_idx.Y + upd_s_idx2])  #
        eucl_dist = sqrt(distance_x ** 2 + distance_y ** 2)
        slack_coll = z[params.i_idx.Slack_Coll]
        v4 = -eucl_dist + minSafetyDistance - slack_coll
        v = np.append(v, np.array([v4]))
    elif n > 2:
        for kk in range(coll_constraints - 1):
            for jj in range(kk + 1, coll_constraints):
                upd_s_idx1 = kk * params.n_states + (n - 1) * params.n_inputs
                upd_s_idx2 = jj * params.n_states + (n - 1) * params.n_inputs
                upd_i_idx = kk * params.n_inputs
                distance_x = (z[params.s_idx.X + upd_s_idx1] - z[params.s_idx.X + upd_s_idx2])
                distance_y = (z[params.s_idx.Y + upd_s_idx1] - z[params.s_idx.Y + upd_s_idx2])
                slack_coll = z[params.i_idx.Slack_Coll + upd_i_idx]
                v4 = -sqrt(distance_x ** 2 + distance_y ** 2) + minSafetyDistance - slack_coll
                v = np.append(v, np.array([v4]))
    return v


nlconst_carN = []
for i in range(5):
    nlconst_carN.append(partial(_nlconst_carN, n=i))
