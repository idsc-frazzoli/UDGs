from bspline.bspline import *
from .car_util import *
from casadi import *

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
    maxspeed = p[params.p_idx.maxspeed]
    targetspeed = p[params.p_idx.targetspeed]
    pdotbeta = p[params.p_idx.pdotbeta]
    plag = p[params.p_idx.plag]
    plat = p[params.p_idx.plat]
    pLeftLane = p[params.p_idx.pLeftLane]
    pab = p[params.p_idx.pab]
    pspeedcostA = p[params.p_idx.pspeedcostA]
    pspeedcostB = p[params.p_idx.pspeedcostB]
    pspeedcostM = p[params.p_idx.pspeedcostM]
    pslack = p[params.p_idx.pslack]

    # get the fancy spline
    splx, sply = casadiDynamicBSPLINE(z[params.s_idx.s], points)
    spldx, spldy = casadiDynamicBSPLINEforward(z[params.s_idx.s], points)
    splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.s], points)
    r = casadiDynamicBSPLINERadius(z[params.s_idx.s], radii)

    forward = vertcat(spldx, spldy)
    sidewards = vertcat(splsx, splsy)

    realPos = vertcat(z[params.s_idx.x], z[params.s_idx.y])
    centerPos = realPos

    wantedpos = vertcat(splx, sply)
    wantedpos_CL = vertcat(splx, sply) + r/2*sidewards
    # todo clarify what is this cost function
    error = centerPos - wantedpos
    error_CL = centerPos - wantedpos_CL
    lagerror = mtimes(forward.T, error)
    laterror = mtimes(sidewards.T, error)
    laterror_CL = mtimes(sidewards.T, error_CL)
    speedcostA = speedPunisherA(z[params.s_idx.vx], targetspeed) * pspeedcostA
    speedcostB = speedPunisherB(z[params.s_idx.vx], targetspeed) * pspeedcostB
    speedcostM = speedPunisherA(z[params.s_idx.vx], maxspeed) * pspeedcostM
    slack = z[params.i_idx.slack]
    lagcost = plag * lagerror ** 2
    leftLaneCost = pLeftLane * laterrorPunisher(laterror, 0)
    latcostCL = plat * laterror_CL ** 2
    regAB = z[params.i_idx.dAb] ** 2 * pab
    regBeta = z[params.i_idx.dBeta] ** 2 * pdotbeta
    return lagcost + leftLaneCost + latcostCL + regAB + regBeta + speedcostA + speedcostB + speedcostM + pslack * slack

