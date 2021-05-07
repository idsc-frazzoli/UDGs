from functools import partial

from .car_util import *
from udgs.bspline.bspline import *
from casadi import *
import numpy as np

from udgs.models import x_idx, u_idx


def _nlconst_car(z, p, n):
    """
    Computes the nonlinear inequality constraints.

    :param z: States and inputs of the system dynamics
    :param p: Parameters of the system dynamics
    :param n: Number of players
    :return: List containing the constraints #todo
    """

    pointsO = params.n_opt_param
    pointsN = params.n_bspline_points

    v = []
    coll_constraints = int(n * (n - 1) / 2)
    minSafetyDistance = p[p_idx.minSafetyDistance]

    for k in range(n):
        upd_s_idx = k * params.n_states + (n - 1) * params.n_inputs
        upd_i_idx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN * 3, pointsN)
        radii = getRadiiFromParameters(p, pointsO + k * pointsN * 3, pointsN)

        splx, sply = casadiDynamicBSPLINE(z[x_idx.S + upd_s_idx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(z[x_idx.S + upd_s_idx], points)
        r = casadiDynamicBSPLINERadius(z[x_idx.S + upd_s_idx], radii)

        wantedpos = vertcat(splx, sply)
        sidewards = vertcat(splsx, splsy)

        realPos = np.array([z[x_idx.X + upd_s_idx], z[x_idx.Y + upd_s_idx]])
        centerPos = realPos
        error = centerPos - wantedpos
        laterror = mtimes(sidewards.T, error)

        slack = z[u_idx.Slack_Lat + upd_i_idx]

        v1 = laterror - r - slack
        v2 = -laterror - r - slack

        v = np.append(v, np.array([v1, v2]))

    if n == 2:
        upd_s_idx1 = params.n_inputs
        upd_s_idx2 = params.n_states + params.n_inputs
        #  upd_i_idx = params.n_inputs
        distance_x = (z[x_idx.X + upd_s_idx1] - z[x_idx.X + upd_s_idx2])  #
        distance_y = (z[x_idx.Y + upd_s_idx1] - z[x_idx.Y + upd_s_idx2])  #
        eucl_dist = sqrt(distance_x ** 2 + distance_y ** 2)
        slack_coll = z[u_idx.Slack_Coll]
        v4 = -eucl_dist + minSafetyDistance - slack_coll
        v = np.append(v, np.array([v4]))
    elif n > 2:
        for kk in range(n - 1):
            for jj in range(kk + 1, n):
                upd_s_idx1 = kk * params.n_states + (n - 1) * params.n_inputs
                upd_s_idx2 = jj * params.n_states + (n - 1) * params.n_inputs
                upd_i_idx = kk * params.n_inputs
                distance_x = (z[x_idx.X + upd_s_idx1] - z[x_idx.X + upd_s_idx2])
                distance_y = (z[x_idx.Y + upd_s_idx1] - z[x_idx.Y + upd_s_idx2])
                slack_coll = z[u_idx.Slack_Coll + upd_i_idx]
                v4 = -sqrt(distance_x ** 2 + distance_y ** 2) + minSafetyDistance - slack_coll
                v = np.append(v, np.array([v4]))

    for jj in range(n):
        upd_s_idx1 = jj * params.n_states + (n - 1) * params.n_inputs
        upd_i_idx = jj * params.n_inputs
        distance_x = (z[x_idx.X + upd_s_idx1] - p[p_idx.Xobstacle])
        distance_y = (z[x_idx.Y + upd_s_idx1] - p[p_idx.Yobstacle])
        slack_obs = z[u_idx.Slack_Obs + upd_i_idx]
        v5 = -sqrt(distance_x ** 2 + distance_y ** 2) + minSafetyDistance - slack_obs
        v = np.append(v, np.array([v5]))
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
    minSafetyDistance = p[p_idx.minSafetyDistance]
    pointsO = params.n_opt_param
    pointsN = params.n_bspline_points
    v = []
    totslackcost = 0
    totlatcost = 0

    for k in range(n):
        upd_s_idx = k * params.n_states + (n - 1) * params.n_inputs
        upd_i_idx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN * 3, pointsN)
        radii = getRadiiFromParameters(p, pointsO + k * pointsN * 3, pointsN)

        splx, sply = casadiDynamicBSPLINE(z[x_idx.S + upd_s_idx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(z[x_idx.S + upd_s_idx], points)
        r = casadiDynamicBSPLINERadius(z[x_idx.S + upd_s_idx], radii)

        wantedpos = vertcat(splx, sply)
        sidewards = vertcat(splsx, splsy)

        realPos = np.array([z[x_idx.X + upd_s_idx], z[x_idx.Y + upd_s_idx]])
        centerPos = realPos
        error = centerPos - wantedpos
        laterror = mtimes(sidewards.T, error)

        slack = z[u_idx.Slack_Lat + upd_i_idx]

        v1 = laterror - r - slack
        v2 = -laterror - r - slack
        v3 = -z[x_idx.S + upd_s_idx] + p[p_idx.TargetProg]

        v = np.append(v, np.array([v1, v2, v3]))

        totslackcost = totslackcost + z[x_idx.CumSlackCost + upd_s_idx]
        totlatcost = totlatcost + z[x_idx.CumLatSpeedCost + upd_s_idx]

    v4 = totslackcost - p[p_idx.OptCost1]
    v5 = totlatcost - p[p_idx.OptCost2]
    v = np.append(v, np.array([v4]))
    v = np.append(v, np.array([v5]))

    if n == 2:
        upd_s_idx1 = params.n_inputs
        upd_s_idx2 = params.n_states + params.n_inputs
        # upd_i_idx = params.n_inputs
        distance_x = (z[x_idx.X + upd_s_idx1] - z[x_idx.X + upd_s_idx2])  #
        distance_y = (z[x_idx.Y + upd_s_idx1] - z[x_idx.Y + upd_s_idx2])  #
        eucl_dist = sqrt(distance_x ** 2 + distance_y ** 2)
        slack_coll = z[u_idx.Slack_Coll]
        v6 = -eucl_dist + minSafetyDistance - slack_coll
        v = np.append(v, np.array([v6]))
    elif n > 2:
        for kk in range(n - 1):
            for jj in range(kk + 1, n):
                upd_s_idx1 = kk * params.n_states + (n - 1) * params.n_inputs
                upd_s_idx2 = jj * params.n_states + (n - 1) * params.n_inputs
                upd_i_idx = kk * params.n_inputs
                distance_x = (z[x_idx.X + upd_s_idx1] - z[x_idx.X + upd_s_idx2])
                distance_y = (z[x_idx.Y + upd_s_idx1] - z[x_idx.Y + upd_s_idx2])
                slack_coll = z[u_idx.Slack_Coll + upd_i_idx]
                v6 = -sqrt(distance_x ** 2 + distance_y ** 2) + minSafetyDistance - slack_coll
                v = np.append(v, np.array([v6]))

    for jj in range(n):
        upd_s_idx1 = jj * params.n_states + (n - 1) * params.n_inputs
        upd_i_idx = jj * params.n_inputs
        distance_x = (z[x_idx.X + upd_s_idx1] - p[p_idx.Xobstacle])
        distance_y = (z[x_idx.Y + upd_s_idx1] - p[p_idx.Yobstacle])
        slack_obs = z[u_idx.Slack_Obs + upd_i_idx]
        v7 = -sqrt(distance_x ** 2 + distance_y ** 2) + minSafetyDistance - slack_obs
        v = np.append(v, np.array([v7]))
    return v


nlconst_carN = []
for i in range(5):
    nlconst_carN.append(partial(_nlconst_carN, n=i))


def _nlconst_car_ibr(z, p, n):
    """
    Computes the nonlinear inequality constraints.

    :param z: States and inputs of the system dynamics
    :param p: Parameters of the system dynamics
    :param n: Number of players
    :return: List containing the constraints #todo
    """

    pointsO = params.n_opt_param
    pointsN = params.n_bspline_points

    minSafetyDistance = p[p_idx.minSafetyDistance]

    points = getPointsFromParameters(p, pointsO, pointsN)
    radii = getRadiiFromParameters(p, pointsO, pointsN)

    splx, sply = casadiDynamicBSPLINE(z[x_idx.S], points)
    splsx, splsy = casadiDynamicBSPLINEsidewards(z[x_idx.S], points)
    r = casadiDynamicBSPLINERadius(z[x_idx.S], radii)

    wantedpos = vertcat(splx, sply)
    sidewards = vertcat(splsx, splsy)

    realPos = np.array([z[x_idx.X], z[x_idx.Y]])
    centerPos = realPos
    error = centerPos - wantedpos
    laterror = mtimes(sidewards.T, error)

    slack = z[u_idx.Slack_Lat]

    # non-collision constraint with an obstacle
    distance_x = (z[x_idx.X] - p[p_idx.Xobstacle])
    distance_y = (z[x_idx.Y] - p[p_idx.Yobstacle])
    slack_obs = z[u_idx.Slack_Obs]

    v1 = laterror - r - slack
    v2 = -laterror - r - slack
    v3 = -sqrt(distance_x ** 2 + distance_y ** 2) + minSafetyDistance - slack_obs

    v = np.array([v1, v2, v3])

    # non-collision constraints with other vehicles

    for i in range(n - 1):
        update_pidx_x = i * 2 + params.n_opt_param + 3 * params.n_bspline_points
        update_pidx_y = i * 2 + 1 + params.n_opt_param + 3 * params.n_bspline_points
        distance_x = (z[x_idx.X] - p[update_pidx_x])
        distance_y = (z[x_idx.Y] - p[update_pidx_y])
        eucl_dist = sqrt(distance_x ** 2 + distance_y ** 2)
        slack_coll = z[u_idx.Slack_Coll]
        v4 = -eucl_dist + minSafetyDistance - slack_coll
        v = np.append(v, np.array([v4]))

    return v


nlconst_car_ibr = []
for i in range(5):
    nlconst_car_ibr.append(partial(_nlconst_car_ibr, n=i))


def _nlconst_car_ibrN(z, p, n):
    """
    Computes the nonlinear inequality constraints.

    :param z: States and inputs of the system dynamics
    :param p: Parameters of the system dynamics
    :return: List containing the constraints #todo
    """
    minSafetyDistance = p[p_idx.minSafetyDistance]
    pointsO = params.n_opt_param
    pointsN = params.n_bspline_points

    points = getPointsFromParameters(p, pointsO, pointsN)
    radii = getRadiiFromParameters(p, pointsO, pointsN)

    splx, sply = casadiDynamicBSPLINE(z[x_idx.S], points)
    splsx, splsy = casadiDynamicBSPLINEsidewards(z[x_idx.S], points)
    r = casadiDynamicBSPLINERadius(z[x_idx.S], radii)

    wantedpos = vertcat(splx, sply)
    sidewards = vertcat(splsx, splsy)

    realPos = np.array([z[x_idx.X], z[x_idx.Y]])
    centerPos = realPos
    error = centerPos - wantedpos
    laterror = mtimes(sidewards.T, error)

    slack = z[u_idx.Slack_Lat]

    # non-collision constraint with an obstacle
    distance_x = (z[x_idx.X] - p[p_idx.Xobstacle])
    distance_y = (z[x_idx.Y] - p[p_idx.Yobstacle])
    slack_obs = z[u_idx.Slack_Obs]

    v1 = laterror - r - slack
    v2 = -laterror - r - slack
    v3 = -z[x_idx.S] + p[p_idx.TargetProg]
    v4 = z[x_idx.CumSlackCost] - p[p_idx.OptCost1]
    v5 = z[x_idx.CumLatSpeedCost] - p[p_idx.OptCost2]
    v6 = -sqrt(distance_x ** 2 + distance_y ** 2) + minSafetyDistance - slack_obs

    v = np.array([v1, v2, v3, v4, v5, v6])

    for i in range(n - 1):
        update_pidx_x = i * 2 + params.n_opt_param + 3 * params.n_bspline_points
        update_pidx_y = i * 2 + 1 + params.n_opt_param + 3 * params.n_bspline_points
        distance_x = (z[x_idx.X] - p[update_pidx_x])
        distance_y = (z[x_idx.Y] - p[update_pidx_y])
        eucl_dist = sqrt(distance_x ** 2 + distance_y ** 2)
        slack_coll = z[u_idx.Slack_Coll]
        v7 = -eucl_dist + minSafetyDistance - slack_coll
        v = np.append(v, np.array([v7]))

    return v


nlconst_car_ibrN = []
for i in range(5):
    nlconst_car_ibrN.append(partial(_nlconst_car_ibrN, n=i))
