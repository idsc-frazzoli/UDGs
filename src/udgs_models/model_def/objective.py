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
    speed_limit = p[params.p_idx.SpeedLimit]
    target_speed = p[params.p_idx.TargetSpeed]
    kReg_dDelta = p[params.p_idx.kReg_dDelta]
    kLag = p[params.p_idx.kLag]
    kLat = p[params.p_idx.kLat]
    pLeftLane = p[params.p_idx.pLeftLane]
    kReg_dAb = p[params.p_idx.kReg_dAb]
    kAboveTargetSpeedCost = p[params.p_idx.kAboveTargetSpeedCost]
    kBelowTargetSpeedCost = p[params.p_idx.kBelowTargetSpeedCost]
    kAboveSpeedLimit = p[params.p_idx.kAboveSpeedLimit]
    kSlack = p[params.p_idx.kSlack]
    pointsO = params.n_param
    pointsN = params.n_bspline_points
    obj = 0

    for k in range(n):
        upd_s_idx = k * params.n_states + (n - 1) * params.n_inputs
        upd_i_idx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN * 3, pointsN)  # todo check indices
        radii = getRadiiFromParameters(p, pointsO + k * pointsN * 3, pointsN)  # todo check indices

        # get the fancy spline
        splx, sply = casadiDynamicBSPLINE(z[params.s_idx.S + upd_s_idx], points)
        spldx, spldy = casadiDynamicBSPLINEforward(z[params.s_idx.S + upd_s_idx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(z[params.s_idx.S + upd_s_idx], points)
        r = casadiDynamicBSPLINERadius(z[params.s_idx.S + upd_s_idx], radii)

        forward = vertcat(spldx, spldy)
        sidewards = vertcat(splsx, splsy)

        realPos = vertcat(z[params.s_idx.X + upd_s_idx], z[params.s_idx.Y + upd_s_idx])
        centerPos = realPos

        wantedpos = vertcat(splx, sply)
        wantedpos_CL = vertcat(splx, sply) + r / 2 * sidewards
        # todo clarify what is this cost function
        error = centerPos - wantedpos
        error_CL = centerPos - wantedpos_CL
        lagerror = mtimes(forward.T, error)
        laterror = mtimes(sidewards.T, error)
        laterror_CL = mtimes(sidewards.T, error_CL)
        speedcostA = speedPunisherA(z[params.s_idx.Vx + upd_s_idx], target_speed) * kAboveTargetSpeedCost
        speedcostB = speedPunisherB(z[params.s_idx.Vx + upd_s_idx], target_speed) * kBelowTargetSpeedCost
        speedcostM = speedPunisherA(z[params.s_idx.Vx + upd_s_idx], speed_limit) * kAboveSpeedLimit
        slack = z[params.i_idx.Slack_Lat + upd_i_idx]
        slackcoll = z[params.i_idx.Slack_Coll + upd_i_idx]
        lagcost = kLag * lagerror ** 2
        leftLaneCost = pLeftLane * laterrorPunisher(laterror, 0)
        latcostCL = kLat * laterror_CL ** 2
        regAB = z[params.i_idx.dAcc + upd_i_idx] ** 2 * kReg_dAb
        regBeta = z[params.i_idx.dDelta + upd_i_idx] ** 2 * kReg_dDelta
        obj = obj + lagcost + leftLaneCost + latcostCL + regAB + regBeta + speedcostA + speedcostB + speedcostM + \
              kSlack * slack + kSlack * slackcoll

    return obj


objective_car = []
for i in range(5):
    objective_car.append(partial(_objective_car, n=i))
