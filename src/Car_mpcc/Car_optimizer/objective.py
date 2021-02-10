from functools import partial

from bspline.bspline import *
from .car_util import *
from casadi import *


def _objective_car(z, p, n):
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
    pointsO = params.n_param
    pointsN = params.n_bspline_points
    obj = 0

    for k in range(n):
        update_for_sidx = k * params.n_states + (n - 1) * params.n_inputs
        update_for_iidx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN * 3, pointsN)  # todo check indices
        radii = getRadiiFromParameters(p, pointsO + k * pointsN * 3, pointsN)  # todo check indices

        # get the fancy spline
        splx, sply = casadiDynamicBSPLINE(z[params.s_idx.s + update_for_sidx], points)
        spldx, spldy = casadiDynamicBSPLINEforward(z[params.s_idx.s + update_for_sidx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.s + update_for_sidx], points)
        r = casadiDynamicBSPLINERadius(z[params.s_idx.s + update_for_sidx], radii)

        forward = vertcat(spldx, spldy)
        sidewards = vertcat(splsx, splsy)

        realPos = vertcat(z[params.s_idx.x + update_for_sidx], z[params.s_idx.y + update_for_sidx])
        centerPos = realPos

        wantedpos = vertcat(splx, sply)
        wantedpos_CL = vertcat(splx, sply) + r / 2 * sidewards
        # todo clarify what is this cost function
        error = centerPos - wantedpos
        error_CL = centerPos - wantedpos_CL
        lagerror = mtimes(forward.T, error)
        laterror = mtimes(sidewards.T, error)
        laterror_CL = mtimes(sidewards.T, error_CL)
        speedcostA = speedPunisherA(z[params.s_idx.vx + update_for_sidx], targetspeed) * pspeedcostA
        speedcostB = speedPunisherB(z[params.s_idx.vx + update_for_sidx], targetspeed) * pspeedcostB
        speedcostM = speedPunisherA(z[params.s_idx.vx + update_for_sidx], maxspeed) * pspeedcostM
        slack = z[params.i_idx.slack + update_for_iidx]
        lagcost = plag * lagerror ** 2
        leftLaneCost = pLeftLane * laterrorPunisher(laterror, 0)
        latcostCL = plat * laterror_CL ** 2
        regAB = z[params.i_idx.dAb + update_for_iidx] ** 2 * pab
        regBeta = z[params.i_idx.dBeta + update_for_iidx] ** 2 * pdotbeta
        obj = obj + lagcost + leftLaneCost + latcostCL + regAB + regBeta + speedcostA + speedcostB + speedcostM + pslack * slack

    return obj


objective_car = []
for i in range(5):
    objective_car.append(partial(_objective_car, n=i))
