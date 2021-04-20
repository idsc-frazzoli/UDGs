from dataclasses import dataclass, field
from typing import NewType, Any
import numpy as np

PlayerName = NewType("PlayerName", str)


@dataclass(unsafe_hash=True)
class CarGeometry:
    """todo"""

    l: float
    """Length of the car """
    l1: float  # Dist from C.O.M to front Tire
    l2: float = field(init=False)  # Dist C.O.M to rear Axle
    w1: float  # Distance between front Tires
    w2: float  # Distance between rear Tires
    back2wheel: float = 0.33  # [m]
    "Distance from the rear of the car to the back axle"
    wheel2front: float = 0.53  # [m]
    "Distance from the front axle to the front of the car"
    wheel2border: float = 0.18
    "Side distance between center of the wheel and external frame"

    def __post_init__(self):
        self.l2 = self.l - self.l1

    def get_wheel_positions(self):
        return np.array(
            [
                [self.l1, self.l1, -self.l2, -self.l2],
                [self.w1 / 2, -self.w1 / 2, self.w2 / 2, -self.w2 / 2]
            ]
        )


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
class CarParams:
    geometry: CarGeometry
    front_tires: WheelGeometry
    rear_tires: WheelGeometry
    color: str

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


@dataclass
class VehicleStateInputs:
    x: Any
    y: Any
    theta: Any
    beta: Any
    ab: Any
