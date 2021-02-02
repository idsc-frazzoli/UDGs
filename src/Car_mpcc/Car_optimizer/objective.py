from bspline.bspline import *
from .car_util import *
from casadi import *
import numpy as np


def objective_car(z, p):
    """
    Computes the objective for the solver

    :param z: States and inputs of the model
    :param points:
    :param radii:
    :param vmax:
    :param maxxacc:
    :param steeringreg:
    :param plag:
    :param plat:
    :param pprog:
    :param pab:
    :param pspeedcost:
    :param pslack:
    :param ptv:
    :return: Objective to be minimized by the solver
    """

    points = getPointsFromParameters(p, params.n_param, params.n_bspline_points)
    # print(points)
    radii = getRadiiFromParameters(p, params.n_param, params.n_bspline_points)
    # print(radii)
    vmax = p[params.p_idx.ps]
    maxxacc = p[params.p_idx.pax]
    steeringreg = p[params.p_idx.pbeta]
    plag = p[params.p_idx.plag]
    plat = p[params.p_idx.plat]
    pprog = p[params.p_idx.pprog]
    pab = p[params.p_idx.pab]
    pspeedcost = p[params.p_idx.pspeedcost]
    pslack = p[params.p_idx.pslack]
    ptv = p[params.p_idx.ptv]

    # get the fancy spline
    # l = gk_geometry.l
    splx, sply = casadiDynamicBSPLINE(z[params.s_idx.s], points)
    spldx, spldy = casadiDynamicBSPLINEforward(z[params.s_idx.s], points)
    splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.s], points)
    r = casadiDynamicBSPLINERadius(z[params.s_idx.s], radii)
    # forward = np.array([[spldx, spldy]])
    # sidewards = np.array([[splsx, splsy]])
    forward = vertcat(spldx, spldy)
    sidewards = vertcat(splsx, splsy)

    realPos = vertcat(z[params.s_idx.x], z[params.s_idx.y])
    centerPos = realPos
    # wantedpos = np.array([[splx, sply]])
    wantedpos = vertcat(splx, sply)
    wantedpos_CL = vertcat(splx, sply) + r/2*sidewards
    # todo clarify what is this cost function
    error = centerPos - wantedpos
    error_CL = centerPos - wantedpos_CL
    lagerror = mtimes(forward.T, error)
    laterror = mtimes(sidewards.T, error)
    laterror_CL = mtimes(sidewards.T, error_CL)
    speedcostA = speedPunisher(z[params.s_idx.vx], vmax) * pspeedcost
    speedcostB = fmin(z[params.s_idx.vx] - vmax, 0) ** 2 * pspeedcost# ~max(v-vmax,0);
    speedcostM = fmax(z[params.s_idx.vx] - vmax+1, 0) ** 2 * pspeedcost
    slack = z[params.i_idx.slack]
    # tv = z[params.i_idx.tv]
    lagcost = plag * lagerror ** 2
    leftLaneCost = plat * fmin(laterror, 0) ** 2
    latcostCL = plat * laterror_CL ** 2
    prog = -pprog * z[params.i_idx.ds]
    regAB = z[params.i_idx.dAb] ** 2 * pab
    regBeta=  z[params.i_idx.dBeta] ** 2 * steeringreg
    return lagcost + leftLaneCost + latcostCL + regAB + regBeta + speedcostA + speedcostB + speedcostM + pslack * slack
