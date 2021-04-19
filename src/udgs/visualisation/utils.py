from numbers import Real
from typing import Tuple, Sequence

import numpy as np


def get_transformed_xy(q: np.array, points: Sequence[Tuple[Real, Real]]) -> Tuple[np.array, np.array]:
    car = tuple((x, y, 1) for x, y in points)
    car = np.array(car).T
    points = q @ car
    x = points[0, :]
    y = points[1, :]
    return x, y


def get_steering_angles(alpha):
    left_angle = -0.63 * alpha * alpha * alpha - 0.31 * alpha * alpha + 0.94 * alpha
    right_angle = -0.63 * alpha * alpha * alpha + 0.31 * alpha * alpha + 0.94 * alpha
    return left_angle, right_angle
