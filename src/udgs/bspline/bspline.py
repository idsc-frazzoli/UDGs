from casadi import *
import numpy as np


def _compute_bspline_bases(x, n):
    x = fmax(x, 0)
    x = fmin(x, n - 2)
    # position in basis function
    if isinstance(x, float):
        v = np.zeros((n, 1))
        b = np.zeros((n, 1))
    else:
        v = SX.zeros(n, 1)
        b = SX.zeros(n, 1)

    for i in range(n):
        v[i, 0] = x - i + 2
        vv = v[i, 0]
        if isinstance(vv, float):
            if vv < 0:
                b[i, 0] = 0
            elif vv < 1:
                b[i, 0] = 0.5 * vv ** 2
            elif vv < 2:
                b[i, 0] = 0.5 * (-3 + 6 * vv - 2 * vv ** 2)
            elif vv < 3:
                b[i, 0] = 0.5 * (3 - vv) ** 2
            else:
                b[i, 0] = 0
        else:
            b[i, 0] = if_else(
                vv < 0,
                0,
                if_else(
                    vv < 1,
                    0.5 * vv ** 2,
                    if_else(
                        vv < 2, 0.5 * (-3 + 6 * vv - 2 * vv ** 2), if_else(vv < 3, 0.5 * (3 - vv) ** 2, 0)
                    ),
                ),
            )
    return b


def casadiDynamicBSPLINE(x, points):
    """

    :param x:
    :param points:
    :return:
    """
    n, _ = points.shape
    b = _compute_bspline_bases(x, n)
    xx = mtimes(b.T, points[:, 0])
    yy = mtimes(b.T, points[:, 1])
    return xx, yy


def casadiDynamicBSPLINERadius(x, radii):
    """

    :param x:
    :param radii:
    :return:
    """
    n, _ = radii.shape
    b = _compute_bspline_bases(x, n)
    rr = mtimes(b.T, radii)
    return rr


def casadiDynamicBSPLINEforward(x, points):
    """

    :param x:
    :param points:
    :return:
    """
    n, _ = points.shape
    x = fmax(x, 0)
    x = fmin(x, n - 2)
    # position in basis function
    if isinstance(x, float):
        v = np.zeros((n, 1))
        b = np.zeros((n, 1))
    else:
        v = SX.zeros(n, 1)
        b = SX.zeros(n, 1)

    for i in range(n):
        v[i, 0] = x - i + 2
        vv = v[i, 0]
        if isinstance(vv, float):
            if vv < 0:
                b[i, 0] = 0
            elif vv < 1:
                b[i, 0] = vv
            elif vv < 2:
                b[i, 0] = 3 - 2 * vv
            elif vv < 3:
                b[i, 0] = -3 + vv
            else:
                b[i, 0] = 0
        else:
            b[i, 0] = if_else(
                vv < 0, 0, if_else(vv < 1, vv, if_else(vv < 2, 3 - 2 * vv, if_else(vv < 3, -3 + vv, 0)))
            )
    xx = mtimes(b.T, points[:, 0])
    yy = mtimes(b.T, points[:, 1])
    norm = (xx ** 2 + yy ** 2) ** 0.5
    xx = xx / norm
    yy = yy / norm
    return xx, yy


def casadiDynamicBSPLINEsidewards(x, points):
    """

    :param x:
    :param points:
    :return:
    """
    yy, xx = casadiDynamicBSPLINEforward(x, points)
    yy = -yy
    return xx, yy
