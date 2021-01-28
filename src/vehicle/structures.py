from dataclasses import dataclass, field
from typing import NewType
import numpy as np

GokartName = NewType("GokartName", str)


@dataclass(unsafe_hash=True)
class GokartGeometry:
    """todo"""

    l: float
    """Length of the gokart ()"""
    l1: float  # Dist from C.O.M to front Tire
    l2: float = field(init=False)  # Dist C.O.M to rear Axle
    w1: float  # Distance between front Tires
    w2: float  # Distance between rear Tires
    f1n: float = field(init=False)  # portion of Mass supported by front tires
    f2n: float = field(init=False)  # portion of Mass supported by rear tire
    back2wheel: float = 0.33  # [m]
    "Distance from the rear of the gokart to the back axle"
    wheel2front: float = 0.53  # [m]
    "Distance from the front axle to the front of the kart"
    wheel2border: float = 0.18
    "Side distance between center of the wheel and external frame"

    def __post_init__(self):
        self.l2 = self.l - self.l1
        self.f1n = self.l2 / self.l  # portion of Mass supported by front tires
        self.f2n = self.l1 / self.l

    def get_wheel_positions(self):
        return np.array(
            [
                [self.l1, self.l1, -self.l2, -self.l2],
                [self.w1 / 2, -self.w1 / 2, self.w2 / 2, -self.w2 / 2]
            ]
        )


@dataclass(frozen=True)
class SteeringColumn:
    """steering column"""

    J_steer: float
    """todo inertia?"""
    b_steer: float
    """todo"""
    k_steer: float
    """todo"""


@dataclass(frozen=True)
class Pacejka:
    """todo"""

    B: float
    C: float
    D: float  # gravity acceleration considered


@dataclass(frozen=True)
class WheelGeometry:
    radius: float
    width: float

    def get_outline(self):
        """
        """

        return np.array(
            [
                [self.radius, -self.radius, -self.radius, self.radius, self.radius],
                [-self.width / 2, -self.width / 2, self.width / 2, self.width / 2, -self.width / 2]
            ]
        )


@dataclass(frozen=True)
class Tire:
    pacejka: Pacejka
    geometry: WheelGeometry


@dataclass(frozen=True)
class GokartParams:
    geometry: GokartGeometry
    front_tires: Tire
    rear_tires: Tire
    steering: SteeringColumn

    def get_outline(self):
        """
        Current shape:
        p6-----p5\
        |         p4
        |         p3
        p1-----p2/
        """
        xtri, ytri = (0.35, 0.45)  # x and y for the front triangle
        xback, xfront = -self.geometry.l2 - self.geometry.back2wheel, self.geometry.l1 + self.geometry.wheel2front
        y = self.geometry.w2 / 2 + self.geometry.wheel2border
        return np.array(
            [
                [xback, xfront - xtri, xfront, xfront, xfront - xtri, xback, xback],
                [-y, -y, -y + ytri, y - ytri, y, y, -y],
            ]
        )


# ------------------------ temp


@dataclass(frozen=True)
class GokartState:
    x: float
    y: float
    psi: float
    vx: float
    vy: float
    dpsi: float
    steer_angle: float  # rad


@dataclass(frozen=True)
class GokartCommands:
    throttle_l: float
    throttle_r: float
    brake: float
