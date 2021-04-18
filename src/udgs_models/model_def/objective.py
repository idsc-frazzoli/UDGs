from functools import partial

from bspline.bspline import *
from .car_util import *
from casadi import *


def _objective_car(z, p, n):
    """
    Computes the objective for the solver

    :param z: States and inputs of the model (forces)
    :param p: Parameters vector of the model (forces)
    :param n: number of vehicles
    """
    speed_limit = p[p_idx.SpeedLimit]
    target_speed = p[p_idx.TargetSpeed]
    kReg_dDelta = p[p_idx.kReg_dDelta]
    kLag = p[p_idx.kLag]
    kLat = p[p_idx.kLat]
    pLeftLane = p[p_idx.pLeftLane]
    kReg_dAb = p[p_idx.kReg_dAb]
    kAboveTargetSpeedCost = p[p_idx.kAboveTargetSpeedCost]
    kBelowTargetSpeedCost = p[p_idx.kBelowTargetSpeedCost]
    kAboveSpeedLimit = p[p_idx.kAboveSpeedLimit]
    kSlack = p[p_idx.kSlack]
    pointsO = params.n_opt_param
    pointsN = params.n_bspline_points
    obj = 0

    for k in range(n):
        upd_x_idx = k * params.n_states + (n - 1) * params.n_inputs
        upd_u_idx = k * params.n_inputs

        points = getPointsFromParameters(p, pointsO + k * pointsN * 3, pointsN)  # todo check indices
        radii = getRadiiFromParameters(p, pointsO + k * pointsN * 3, pointsN)  # todo check indices

        # get the fancy spline
        splx, sply = casadiDynamicBSPLINE(z[x_idx.S + upd_x_idx], points)
        spldx, spldy = casadiDynamicBSPLINEforward(z[x_idx.S + upd_x_idx], points)
        splsx, splsy = casadiDynamicBSPLINEsidewards(z[x_idx.S + upd_x_idx], points)
        r = casadiDynamicBSPLINERadius(z[x_idx.S + upd_x_idx], radii)

        forward = vertcat(spldx, spldy)
        sidewards = vertcat(splsx, splsy)

        realPos = vertcat(z[x_idx.X + upd_x_idx], z[x_idx.Y + upd_x_idx])
        centerPos = realPos

        wantedpos = vertcat(splx, sply)
        wantedpos_CL = vertcat(splx, sply) + r / 2 * sidewards
        # todo clarify what is this cost function
        error = centerPos - wantedpos
        error_CL = centerPos - wantedpos_CL
        lagerror = mtimes(forward.T, error)
        laterror = mtimes(sidewards.T, error)
        laterror_CL = mtimes(sidewards.T, error_CL)
        speedcostA = speedPunisherA(z[x_idx.Vx + upd_x_idx], target_speed) * kAboveTargetSpeedCost
        speedcostB = speedPunisherB(z[x_idx.Vx + upd_x_idx], target_speed) * kBelowTargetSpeedCost
        speedcostM = speedPunisherA(z[x_idx.Vx + upd_x_idx], speed_limit) * kAboveSpeedLimit
        slack = z[u_idx.Slack_Lat + upd_u_idx]
        slackcoll = z[u_idx.Slack_Coll + upd_u_idx]
        slackobs = z[u_idx.Slack_Obs + upd_u_idx]
        lagcost = kLag * lagerror ** 2
        leftLaneCost = pLeftLane * laterrorPunisher(laterror, 0)
        latcostCL = kLat * laterror_CL ** 2
        regAB = z[u_idx.dAcc + upd_u_idx] ** 2 * kReg_dAb
        regBeta = z[u_idx.dDelta + upd_u_idx] ** 2 * kReg_dDelta
        obj = obj + lagcost + leftLaneCost + latcostCL + regAB + regBeta + speedcostA + speedcostB + speedcostM + \
              kSlack * slack + kSlack * slackcoll + kSlack * slackobs

    return obj


objective_car = []
for i in range(5):
    objective_car.append(partial(_objective_car, n=i))
