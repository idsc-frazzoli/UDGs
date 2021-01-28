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
    radii = getRadiiFromParameters(p, params.n_param, params.n_bspline_points)
    targetSpeed=p[params.p_idx.targetSpeed]
    maxSpeed = p[params.p_idx.maxSpeed]

    pLag = p[params.p_idx.pLag]
    pLat = p[params.p_idx.pLat]
    pLeftLane = p[params.p_idx.pLeftLane]
    pDotBeta = p[params.p_idx.pDotBeta]
    pab = p[params.p_idx.pab]
    pSpeedCostA = p[params.p_idx.pSpeedCostA]
    pSpeedCostB = p[params.p_idx.pSpeedCostB]
    pSpeedCostM = p[params.p_idx.pSpeedCostM]
    pSlack = p[params.p_idx.pSlack]

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

    wantedpos = vertcat(splx, sply)
    wantedpos_CL = vertcat(splx, sply) + r/2 * sidewards

    error = centerPos - wantedpos
    error_CL = centerPos - wantedpos_CL
    lagerror = mtimes(forward.T, error)
    laterror = mtimes(sidewards.T, error)
    laterror_CL = mtimes(sidewards.T, error_CL)

    speedcostA = speedPunisher(z[params.s_idx.v], targetSpeed) * pSpeedCostA
    speedcostB = speedPunisherB(z[params.s_idx.v], targetSpeed) * pSpeedCostB
    speedcostM = speedPunisher(z[params.s_idx.v], maxSpeed) * pSpeedCostM

    slack = z[params.i_idx.slack]
    lagcost = pLag * lagerror ** 2
    leftLaneCost = pLeftLane * latErrorPunisher(laterror)
    latcostCL = pLat * laterror_CL

    regAB = z[params.i_idx.dAb] ** 2 * pab
    regBeta = z[params.i_idx.dBeta] ** 2 * pDotBeta
    return lagcost + leftLaneCost + latcostCL + regAB + regBeta + speedcostA + speedcostB + speedcostM + pSlack * slack
