from casadi import *
import numpy as np

from udgs_models.model_def import params


def acclim(VELY, VELX, taccx, maxA):
    """

    :param VELY:
    :param VELX:
    :param taccx:
    :param maxA:
    :return:
    """
    return (VELX ** 2 + VELY ** 2) * taccx ** 2 - VELX ** 2 * maxA ** 2


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


def casadiGetSmoothMaxAcc(x):
    """

    :param x:
    :return:
    """
    # used for testing before using it in casadi
    cp0 = 1.9173276271
    cp1 = -0.0113682655
    cp2 = -0.0150793283
    cp3 = 0.0023869979

    cn0 = -1.4265329731
    cn1 = -0.1612157772
    cn2 = 0.0503284643
    cn3 = -0.0048860339

    cp = lambda x: cp0 + cp1 * x + cp2 * x ** 2 + cp3 * x ** 3
    cn = lambda x: cn0 + cn1 * x + cn2 * x ** 2 + cn3 * x ** 3
    si = lambda x: 0.5 + 1.5 * x - 2 * x ** 3

    st = 0.5
    posval = cp(st)
    negval = -cn(st)

    if isinstance(x, float):
        # this is also called with doubles by Forces
        if x > st:
            acc = cp(x)
        elif x > -st:
            acc = negval * (1 - si(x)) + posval * si(x)
        else:
            acc = -cn(-x)
    else:
        acc = if_else(x > st, cp(x), if_else(x > -st, negval * (1 - si(x)) + posval * si(x), -cn(-x)))

    return acc


def casadiGetMaxAcc(x):
    """

    :param x:
    :return:
    """
    # used for testing before using it in casadi
    cp0 = 1.9173276271
    cp1 = -0.0113682655
    cp2 = -0.0150793283
    cp3 = 0.0023869979

    cn0 = -1.4265329731
    cn1 = -0.1612157772
    cn2 = 0.0503284643
    cn3 = -0.0048860339

    cp = lambda x: cp0 + cp1 * x + cp2 * x ** 2 + cp3 * x ** 3
    cn = lambda x: cn0 + cn1 * x + cn2 * x ** 2 + cn3 * x ** 3

    st = 0.5
    posval = cp(st)
    negval = -cn(st)

    if isinstance(x, float):
        # this is also called with doubles by Forces
        if x > st:
            acc = cp(x)
        elif x > -st:
            acc = (x + st) / (2 * st) * (posval - negval) + negval
        else:
            acc = -cn(-x)
    else:
        acc = if_else(
            x > st, cp(x), if_else(x > -st, (x + st) / (2 * st) * (posval - negval) + negval, -cn(-x))
        )

    return acc


def speedPunisherA(v, vmax):
    """

    :param v:
    :param vmax:
    :return:
    """
    x = fmax(v - vmax, 0)
    return x ** 2

def speedPunisherB(v, vmax):
    """

    :param v:
    :param vmax:
    :return:
    """
    x = fmin(v - vmax, 0)
    return x ** 2

def laterrorPunisher(laterror,cc):
    """

    :param laterror:
    :param cc:
    :return:
    """
    x = fmin(laterror, 0)
    return x ** 2

def set_p_car(
    maxspeed,
    targetspeed,
    optcost1,
    optcost2,
    Xobstacle,
    Yobstacle,
    targetprog,
    pspeedcostA,
    pspeedcostB,
    pspeedcostM,
    plag,
    plat,
    pLeftLane,
    pab,
    pdotbeta,
    pslack,
    distance,
    carLength,
    points,
    num_cars):
    p = np.zeros((params.n_param + 3 * params.n_bspline_points * num_cars))
    p[params.p_idx.SpeedLimit] = maxspeed
    p[params.p_idx.TargetSpeed] = targetspeed
    p[params.p_idx.OptCost1] = optcost1
    p[params.p_idx.OptCost2] = optcost2
    p[params.p_idx.Xobstacle] = Xobstacle
    p[params.p_idx.Yobstacle] = Yobstacle
    p[params.p_idx.TargetProg] = targetprog
    p[params.p_idx.kAboveTargetSpeedCost] = pspeedcostA
    p[params.p_idx.kBelowTargetSpeedCost] = pspeedcostB
    p[params.p_idx.kAboveSpeedLimit] = pspeedcostM
    p[params.p_idx.kLag] = plag
    p[params.p_idx.kLat] = plat
    p[params.p_idx.pLeftLane] = pLeftLane
    p[params.p_idx.kReg_dAb] = pab
    p[params.p_idx.kReg_dDelta] = pdotbeta
    p[params.p_idx.kSlack] = pslack
    p[params.p_idx.minSafetyDistance] = distance
    p[params.p_idx.carLength] = carLength
    for k in range(num_cars):
        update = 3 * params.n_bspline_points * k
        temp = points[params.n_bspline_points * k: params.n_bspline_points + params.n_bspline_points * k, :]
        p[params.n_param + update: params.n_param + 3 * params.n_bspline_points + update] = temp.flatten(order="f")

    return p
