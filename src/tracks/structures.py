from dataclasses import dataclass
from numbers import Number
from typing import Sequence, Optional
import numpy as np


@dataclass
class SplineTrack:
    x: Sequence[Number]
    y: Sequence[Number]
    radius: Sequence[Number]

    def __post_init__(self):
        assert len(set(map(len, [self.x, self.y, self.radius]))) == 1

    def as_np_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.radius]).T

    def get_control_point(self, idx: int):
        idx %= len(self.x)
        return np.array([self.x[idx], self.y[idx], self.radius[idx]])


@dataclass
class Track:
    desc: str
    spline: SplineTrack
    background: Optional[np.ndarray]
    "Background image, usually an occupancy grid"
    scale_factor: Optional[float]
    "Scale factor as meters/pixels of the background"


