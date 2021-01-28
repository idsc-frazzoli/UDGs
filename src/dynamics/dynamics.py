from functools import partial
from .dynamics_utils import *
import numpy as np

from vehicle import KITT, gokart_pool
from vehicle.structures import GokartGeometry

__all__ = ["kitt_dynamics"]


def _dynamics(VELX, VELY, VELROTZ, BETA, AB, TV, B1, C1, D1, B2, C2, D2, Ic, gk_geometry: GokartGeometry):
    """
    Calculates accelerations of the Cart based on the slipping
    Tricyclemodeldescribed in MarcHeimsMastersThesis2019
    N.B.ThisFunction is Massinvariant as alltheforcesappliedarefactorsof
    theNormalcontractforcesothemasscancelsoutofallequations

    :param VELX:
    :param VELY:
    :param VELROTZ:
    :param BETA: Lenkwinkel (control variable)
    :param AB: acceleration of hinterachse (controlvariable)
    :param TV: torquevectoring(control variable) AB - TV rechte achse AB + TV linke achse
    :param B1: Front Tire Params(for magic formula)
    :param C1: Front Tire Params(for magic formula)
    :param D1: Front Tire Params(for magic formula)
    :param B2: Rear Tire Params(for magic formula)
    :param C2: Rear Tire Params(for magic formula)
    :param D2: Rear Tire Params(for magic formula)
    :param Ic: Moment of inertia
    :return:
    """
    # modelDx:
    # .
    # HoweverThemagicformulaco - efs(param[1:6])maychangewith mass

    reg = 0.5  # fixme what the hell is this?
    # Tire Forces (velocity in wheels reference frame)
    vel1 = mtimes(rotmat(BETA).T, np.array([[VELX], [VELY + gk_geometry.l1 * VELROTZ]]))
    # lateral acceleration on front wheels in wheels ref frame
    f1y = simplefaccy(vel1[1], vel1[0], reg=reg, B1=B1, C1=C1, D1=D1)
    # accelerations on front wheel in cart ref frame
    F1 = mtimes(rotmat(-BETA).T, np.array([[0], [f1y]])) * gk_geometry.f1n
    # forward acceleration on cart from front wheels
    F1x = F1[0]
    # Lateral acceleration on cart from front wheels
    F1y = F1[1]

    F2x = AB
    vely = VELY - gk_geometry.l2 * VELROTZ
    f2n = gk_geometry.f2n
    # Lateral acceleration from from right rear wheel
    F2y1 = simpleaccy(vely, VELX, (AB + TV / 2) / f2n, reg=reg, B2=B2, C2=C2, D2=D2) * f2n / 2  # fixme faccy?
    # Lateral acceleration from from left rear wheel
    F2y2 = simpleaccy(vely, VELX, (AB - TV / 2) / f2n, reg=reg, B2=B2, C2=C2, D2=D2) * f2n / 2
    # Lateral acceleration from rear wheels
    F2y = simpleaccy(vely, VELX, AB / f2n, reg=reg, B2=B2, C2=C2, D2=D2) * f2n
    TVTrq = TV * gk_geometry.w2  # Torque from difference in real wheel accelerations

    # # Cart Accelerations
    ACCROTZ = (
        TVTrq + F1y * gk_geometry.l1 - F2y * gk_geometry.l2
    ) / Ic  # Rotational Acceleration of the cart
    ACCX = F1x + F2x + VELROTZ * VELY  # Forward Acceleration of cart
    ACCY = F1y + F2y1 + F2y2 - VELROTZ * VELX  # Lateral Acceleration of cart
    return ACCX, ACCY, ACCROTZ


kitt_dynamics = partial(_dynamics, gk_geometry=gokart_pool[KITT].geometry)
