from casadi import *
import numpy as np


def rotmat(alpha):
    return np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])


def magic(s, B, C, D):
    """ pajecka"""
    return D * sin(C * atan(B * s))


def capfactor(taccx, D2):
    return sqrt(1 - satfun(pow(taccx / D2, 2)))


def satfun(x):
    l = 0.8
    r = 1 - l
    if isinstance(x, float):
        if x < l:
            y = x
        elif x < 1 + r:
            d = (1 + r - x) / r
            y = 1 - 1 / 4 * r * d ** 2
        else:
            y = 1
        y = 0.95 * y
    else:
        d = (1 + r - x) / r
        y = 0.95 * if_else(x < l, x, if_else(x < 1 + r, 1 - 1 / 4 * r * d ** 2, 1))
    return y


def simpleslip(VELY, VELX, taccx, reg, D2):
    return -(1 / capfactor(taccx, D2=D2)) * VELY / (VELX + reg)


def simpleaccy(VELY, VELX, taccx, reg, B2, C2, D2):
    """force / kg   applied  by  rear tires   with Taccx travelling at Velx, Vely"""
    accy = magic(simpleslip(VELY=VELY, VELX=VELX, taccx=taccx, reg=reg, D2=D2), B2, C2, D2)
    return capfactor(taccx, D2=D2) * accy


def simplefaccy(VELY, VELX, reg, B1, C1, D1):
    """force/kg applied by front tire travelling at Velx,Vely"""
    return magic(-VELY / (VELX + reg), B1, C1, D1)


def ackermann_map(alpha):
    """ expected value between -1 and +1 """
    return -0.25 * alpha * alpha * alpha + 0.71 * alpha
