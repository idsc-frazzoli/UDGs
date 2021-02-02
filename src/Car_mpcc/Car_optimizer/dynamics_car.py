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
    # tv = u[params.i_idx.tv]
    dotbeta = u[params.i_idx.dBeta]
    ds = u[params.i_idx.ds]

    theta = x[params.s_idx.theta - params.n_inputs]
    vx = x[params.s_idx.vx - params.n_inputs]
    # vy = x[params.s_idx.vy - params.n_inputs]
    dtheta = x[params.s_idx.dtheta - params.n_inputs]
    beta = x[params.s_idx.beta - params.n_inputs]  # from steering.
    # ackermannAngle = ackermann_map(beta)  # ackermann Mapping

    # acc_x, acc_y, rotacc_z = kitt_dynamics(
    #     VELX=vx,
    #     VELY=vx,
    #     VELROTZ=dtheta,
    #     BETA=beta,
    #     AB=ab,
    #     TV=tv,
    #     B1=p[params.p_idx.pacFB],
    #     C1=p[params.p_idx.pacFC],
    #     D1=p[params.p_idx.pacFD],
    #     B2=p[params.p_idx.pacRB],
    #     C2=p[params.p_idx.pacRC],
    #     D2=p[params.p_idx.pacRD],
    #     Ic=p[params.p_idx.pmoi],
    # )

    if isinstance(x[0], float):
        dx = np.zeros(params.n_states)
    else:  # fixme not sure what this was for
        dx = SX.zeros(params.n_states, 1)
    lc = p[params.p_idx.carLength]
    # rot_mat = rotmat(theta)
    # lv = np.array([[vx], [vy]])
    # gv = mtimes(rot_mat, lv)
    dx[params.s_idx.x - params.n_inputs] = vx * cos(theta)
    dx[params.s_idx.y - params.n_inputs] = vx * sin(theta)
    dx[params.s_idx.theta - params.n_inputs] = vx * tan(beta) / lc
    dx[params.s_idx.vx - params.n_inputs] = ab
    dx[params.s_idx.ab - params.n_inputs] = dotab
    dx[params.s_idx.beta - params.n_inputs] = dotbeta
    dx[params.s_idx.s - params.n_inputs] = ds
    return dx
