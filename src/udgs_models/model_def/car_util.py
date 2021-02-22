from casadi import *
import numpy as np

from udgs_models.model_def import *


def getPointsFromParameters(p, pointsO, pointsN):
    """

    :param p:
    :param pointsO:
    :param pointsN:
    :return:
    """
    data = p[pointsO: pointsO + pointsN * 2]  # todo check indices
    return reshape(data, (pointsN, 2))


def getRadiiFromParameters(p, pointsO, pointsN):
    """

    :param p:
    :param pointsO:
    :param pointsN:
    :return:
    """
    return p[pointsO + pointsN * 2: pointsO + pointsN * 3]


def speedPunisherA(v, vmax):
    """

    :param v:
    :param vmax:
    :return:
    """
    x = fmax(v - vmax, 0)
    return x ** 2


def speedPunisherB(v, vmin):
    """

    :param v:
    :param vmin:
    :return:
    """
    x = fmin(v - vmin, 0)
    return x ** 2


def laterrorPunisher(laterror, cc):
    """

    :param laterror:
    :param cc:
    :return:
    """
    x = fmin(laterror, 0)
    return x ** 2


def set_p_car(
        SpeedLimit,
        TargetSpeed,
        OptCost1,
        OptCost2,
        Xobstacle,
        Yobstacle,
        TargetProg,
        kAboveTargetSpeedCost,
        kBelowTargetSpeedCost,
        kAboveSpeedLimit,
        kLag,
        kLat,
        pLeftLane,
        kReg_dAb,
        kReg_dDelta,
        kSlack,
        minSafetyDistance,
        carLength,
        points,
        n_players):
    p = np.zeros((params.n_opt_param + 3 * params.n_bspline_points * n_players))
    p[p_idx.SpeedLimit] = SpeedLimit
    p[p_idx.TargetSpeed] = TargetSpeed
    p[p_idx.OptCost1] = OptCost1
    p[p_idx.OptCost2] = OptCost2
    p[p_idx.Xobstacle] = Xobstacle
    p[p_idx.Yobstacle] = Yobstacle
    p[p_idx.TargetProg] = TargetProg
    p[p_idx.kAboveTargetSpeedCost] = kAboveTargetSpeedCost
    p[p_idx.kBelowTargetSpeedCost] = kBelowTargetSpeedCost
    p[p_idx.kAboveSpeedLimit] = kAboveSpeedLimit
    p[p_idx.kLag] = kLag
    p[p_idx.kLat] = kLat
    p[p_idx.pLeftLane] = pLeftLane
    p[p_idx.kReg_dAb] = kReg_dAb
    p[p_idx.kReg_dDelta] = kReg_dDelta
    p[p_idx.kSlack] = kSlack
    p[p_idx.minSafetyDistance] = minSafetyDistance
    p[p_idx.carLength] = carLength
    for k in range(n_players):
        update = 3 * params.n_bspline_points * k
        temp = points[params.n_bspline_points * k: params.n_bspline_points + params.n_bspline_points * k, :]
        p[params.n_opt_param + update: params.n_opt_param + 3 * params.n_bspline_points + update] = \
            temp.flatten(order="f")

    return p


def set_p_car_ibr(
        SpeedLimit,
        TargetSpeed,
        OptCost1,
        OptCost2,
        Xobstacle,
        Yobstacle,
        TargetProg,
        kAboveTargetSpeedCost,
        kBelowTargetSpeedCost,
        kAboveSpeedLimit,
        kLag,
        kLat,
        pLeftLane,
        kReg_dAb,
        kReg_dDelta,
        kSlack,
        minSafetyDistance,
        carLength,
        points,
        coordinateX_car,
        coordinateY_car,
        n_players):
    num_coordinates_other_players = (n_players - 1) * 2
    opt_params = params.n_opt_param
    p = np.zeros((opt_params + num_coordinates_other_players + 3 * params.n_bspline_points))

    p[p_idx.SpeedLimit] = SpeedLimit
    p[p_idx.TargetSpeed] = TargetSpeed
    p[p_idx.OptCost1] = OptCost1
    p[p_idx.OptCost2] = OptCost2
    p[p_idx.Xobstacle] = Xobstacle
    p[p_idx.Yobstacle] = Yobstacle
    p[p_idx.TargetProg] = TargetProg
    p[p_idx.kAboveTargetSpeedCost] = kAboveTargetSpeedCost
    p[p_idx.kBelowTargetSpeedCost] = kBelowTargetSpeedCost
    p[p_idx.kAboveSpeedLimit] = kAboveSpeedLimit
    p[p_idx.kLag] = kLag
    p[p_idx.kLat] = kLat
    p[p_idx.pLeftLane] = pLeftLane
    p[p_idx.kReg_dAb] = kReg_dAb
    p[p_idx.kReg_dDelta] = kReg_dDelta
    p[p_idx.kSlack] = kSlack
    p[p_idx.minSafetyDistance] = minSafetyDistance
    p[p_idx.carLength] = carLength
    p[opt_params: opt_params + 3 * params.n_bspline_points] = points.flatten(order="f")
    for i in range(n_players-1):
        updateX = 2*i
        updateY = 1 + 2*i
        p[params.n_opt_param + 3 * params.n_bspline_points + updateX] = coordinateX_car
        p[params.n_opt_param + 3 * params.n_bspline_points + updateY] = coordinateY_car

    return p
