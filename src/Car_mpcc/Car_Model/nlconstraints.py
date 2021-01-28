from .car_util import *
from bspline.bspline import *
from casadi import *
import numpy as np


def nlconst_car(z, p):
    """
    Computes the nonlinear inequality constraints.

    :param z: States and inputs of the system dynamics
    :param p: Parameters of the system dynamics
    :return: List containing the constraints #todo
    """

    pointsO = params.n_param
    pointsN = params.n_bspline_points
    points = getPointsFromParameters(p, pointsO, pointsN)
    radii = getRadiiFromParameters(p, pointsO, pointsN)

    splx, sply = casadiDynamicBSPLINE(z[params.s_idx.s], points)
    splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.s], points)
    r = casadiDynamicBSPLINERadius(z[params.s_idx.s], radii)

    wantedpos = vertcat(splx, sply)
    sidewards = vertcat(splsx, splsy)

    realPos = np.array([[z[params.s_idx.x]], [z[params.s_idx.y]]])
    centerPos = realPos
    error = centerPos - wantedpos
    laterror = mtimes(sidewards.T, error)

    v1 = z[params.s_idx.ab] - casadiGetSmoothMaxAcc(z[params.s_idx.v])
    v2 = laterror - r
    v3 = -laterror - r

    v = np.array([v1, v2, v3])
    return v

def nlconst_car_N(z, p):
    """
    Computes the nonlinear inequality constraints at the terminal point.

    :param z: States and inputs of the system dynamics
    :param p: Parameters of the system dynamics
    :return: List containing the constraints #todo
    """

    pointsO = params.n_param
    pointsN = params.n_bspline_points
    points = getPointsFromParameters(p, pointsO, pointsN)
    radii = getRadiiFromParameters(p, pointsO, pointsN)

    splx, sply = casadiDynamicBSPLINE(z[params.s_idx.s], points)
    splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.s], points)
    r = casadiDynamicBSPLINERadius(z[params.s_idx.s], radii)

    wantedpos = vertcat(splx, sply)
    sidewards = vertcat(splsx, splsy)

    realPos = np.array([[z[params.s_idx.x]], [z[params.s_idx.y]]])
    centerPos = realPos
    error = centerPos - wantedpos
    laterror = mtimes(sidewards.T, error)
    progressMax = p[params.p_idx.progressMax]

    v1 = z[params.s_idx.ab] - casadiGetSmoothMaxAcc(z[params.s_idx.v])
    v2 = laterror - r
    v3 = -laterror - r
    v4 = -z[params.s_idx.s] + progressMax

    v = np.array([v1, v2, v3, v4])
    return v
