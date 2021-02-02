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

    #forwardacc = z[params.s_idx.ab]
    #VELX = z[params.s_idx.vx]
    # = z[params.s_idx.vy]
    slack = z[params.i_idx.slack]

    pointsO = params.n_param
    pointsN = params.n_bspline_points
    points = getPointsFromParameters(p, pointsO, pointsN)
    radii = getRadiiFromParameters(p, pointsO, pointsN)

    splx, sply = casadiDynamicBSPLINE(z[params.s_idx.s], points)
    spldx, spldy = casadiDynamicBSPLINEforward(z[params.s_idx.s], points)
    splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.s], points)
    r = casadiDynamicBSPLINERadius(z[params.s_idx.s], radii)

    wantedpos = vertcat(splx, sply)
    forward = vertcat(spldx, spldy)
    sidewards = vertcat(splsx, splsy)

    realPos = np.array([[z[params.s_idx.x]], [z[params.s_idx.y]]])
    # centerOffset = 0.2*gokartforward(z(index.theta))'
    centerPos = realPos
    error = centerPos - wantedpos
    # lagerror = mtimes(forward.T, error)
    laterror = mtimes(sidewards.T, error)

    v1 = z[params.s_idx.ab] - casadiGetSmoothMaxAcc(z[params.s_idx.vx])
    # v2 = z[params.s_idx.ab] - z[params.i_idx.tv] - casadiGetSmoothMaxAcc(z[params.s_idx.vx])
    # v3 = acclim(VELY, VELX, forwardacc, p[params.p_idx.pax]) - slack
    v4 = laterror - r - 0.5 * slack
    v5 = -laterror - r - 0.5 * slack

    v = np.array([v1, v4, v5])
    return v
