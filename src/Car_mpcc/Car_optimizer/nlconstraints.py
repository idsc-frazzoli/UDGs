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

    for k in range(n):
        update_for_sidx = k * params.n_states + (n-1) * params.n_inputs
        update_for_iidx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN, (k + 1) * pointsN)  # todo check indices
        radii = getRadiiFromParameters(p, pointsO + k * pointsN, (k + 1) * pointsN)  # todo check indices

        splx, sply = casadiDynamicBSPLINE(z[params.s_idx.s + update_for_sidx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.s + update_for_sidx], points)
        r = casadiDynamicBSPLINERadius(z[params.s_idx.s + update_for_sidx], radii)

        wantedpos = vertcat(splx, sply)
        sidewards = vertcat(splsx, splsy)

        realPos = np.array([[z[params.s_idx.x + update_for_sidx]], [z[params.s_idx.y + update_for_sidx]]])
        centerPos = realPos
        error = centerPos - wantedpos
        laterror = mtimes(sidewards.T, error)

        slack = z[params.i_idx.slack + update_for_iidx]

        v1 = laterror - r - slack
        v2 = -laterror - r - slack

        v = np.append(np.array([v1, v2]))

    return v


nlconst_car = []
for i in range(7):
    nlconst_car.append(partial(_nlconst_car, n=i))


def _nlconst_carN(z, p, n):
    """
    Computes the nonlinear inequality constraints.

    :param z: States and inputs of the system dynamics
    :param p: Parameters of the system dynamics
    :return: List containing the constraints #todo
    """

    pointsO = params.n_param
    pointsN = params.n_bspline_points
    v = []

    for k in range(n):
        update_for_sidx = k * params.n_states + (n - 1) * params.n_inputs
        update_for_iidx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN, (k + 1) * pointsN)  # todo check indices
        radii = getRadiiFromParameters(p, pointsO + k * pointsN, (k + 1) * pointsN)  # todo check indices

        splx, sply = casadiDynamicBSPLINE(z[params.s_idx.s + update_for_sidx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.s + update_for_sidx], points)
        r = casadiDynamicBSPLINERadius(z[params.s_idx.s + update_for_sidx], radii)

        wantedpos = vertcat(splx, sply)
        sidewards = vertcat(splsx, splsy)

        realPos = np.array([[z[params.s_idx.x + update_for_sidx]], [z[params.s_idx.y + update_for_sidx]]])
        centerPos = realPos
        error = centerPos - wantedpos
        laterror = mtimes(sidewards.T, error)

        slack = z[params.i_idx.slack + update_for_iidx]

        v1 = laterror - r - slack
        v2 = -laterror - r - slack
        v3 = -z[params.s_idx.s + update_for_sidx] + p[params.p_idx.targetprog]

        v = np.append(np.array([v1, v2, v3]))

    return v


nlconst_carN = []
for i in range(7):
    nlconst_carN.append(partial(_nlconst_carN, n=i))
