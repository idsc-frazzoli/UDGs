from dataclasses import dataclass
from numbers import Number
from typing import Sequence, Optional, List
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


class Lane:
    def __init__(self, id: str, control_points: List[List[float]]):
        self.id_ = id
        self.control_points = self._init_control_points(control_points)

    @property
    def id(self):
        return self.id_

    def _init_control_points(self, control_points):
        ctrl_pnts = np.array(control_points).astype(float)
        ctrl_pnts[:, -1] *= np.pi / 180  # deg2rad
        return ctrl_pnts




class Scenario:
    def __init__(self, yaml_dict: dict):
        self.background_: str = yaml_dict["background"]
        self.scale_factor: float = yaml_dict["scale_factor"]
        self.lanes: List[Lane] = self._init_lanes(yaml_dict["lanes"])

    def _init_lanes(self, lanes_yml: dict):
        lanes = []
        for key, elements in lanes_yml.items():
            lanes.append(Lane(id=key, control_points=elements))

        return lanes
