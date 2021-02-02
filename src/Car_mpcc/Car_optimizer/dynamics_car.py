from Car_mpcc.Car_optimizer import params
from casadi import *
import numpy as np


def dynamics_car(x, u, p):
    """todo

    :param x: state variables
    :param u: control inputs variables
    :param p: parameter variables
    :return:
    """
    dotab = u[params.i_idx.dAb]
    ab = x[params.s_idx.ab - params.n_inputs]
    dotbeta = u[params.i_idx.dBeta]
    ds = u[params.i_idx.ds]

    theta = x[params.s_idx.theta - params.n_inputs]
    vx = x[params.s_idx.vx - params.n_inputs]
    beta = x[params.s_idx.beta - params.n_inputs]  # from steering.

    if isinstance(x[0], float):
        dx = np.zeros(params.n_states)
    else:  # fixme not sure what this was for
        dx = SX.zeros(params.n_states, 1)
    lc = p[params.p_idx.carLength]

    dx[params.s_idx.x - params.n_inputs] = vx * cos(theta)
    dx[params.s_idx.y - params.n_inputs] = vx * sin(theta)
    dx[params.s_idx.theta - params.n_inputs] = vx * tan(beta) / lc
    dx[params.s_idx.vx - params.n_inputs] = ab
    dx[params.s_idx.ab - params.n_inputs] = dotab
    dx[params.s_idx.beta - params.n_inputs] = dotbeta
    dx[params.s_idx.s - params.n_inputs] = ds
    dx[params.s_idx.CumSlackCost - params.n_inputs] = 1
    dx[params.s_idx.CumLatSpeedCost - params.n_inputs] = 1
    return dx
