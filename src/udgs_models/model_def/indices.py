from collections import namedtuple
from dataclasses import dataclass
from enum import IntEnum, unique
from math import pi
from typing import Tuple

"""
All indices in the problem formulation are expected to be 0-based in Python, as is usual in this language. 
This does not include the indices of the generated solver, however, where outputs are named x01, x02, … as in MATLAB. 
Thus, the problem formulation before generation requires 0-based indices, whereas the returned solver from the 
server uses 1-based indices. This also does not apply to the low-level Python interface, where indices are 1-based 
even in the model formulation.
"""


# todo description for every parameter


@unique
class IdxInput(IntEnum):
    dAcc = 0
    """desired change of acceleration"""
    dDelta = 1
    """rate of change steering angle"""
    dS = 2
    """Derivative of the progress along the track"""
    Slack_Lat = 3
    """TODO"""
    Slack_Coll = 4
    """TODO"""
    Slack_Obs = 5
    """TODO"""


Pair = Tuple[float, float]


@dataclass(frozen=True)
class InputConstraints:
    dS: Pair = (-1, 5)
    dAcc: Pair = (-10, 10)


input_constraints = InputConstraints()


@unique
class IdxState(IntEnum):
    X = 6
    """x-position in world frame"""
    Y = 7
    """y-position in world frame"""
    Theta = 8
    """orientation in world frame"""
    Vx = 9
    """forward velocity in kart frame"""
    Acc = 10
    """acceleration and braking"""
    Delta = 11
    """steering angle"""
    S = 12
    """path progress in local spline reference system"""
    CumSlackCost = 13
    """cumulative slack"""
    CumLatSpeedCost = 14
    """cumulative left laterror cost + cost for exceeding speed limit"""


@dataclass(frozen=True)
class StateConstraints:
    Acc: Pair = (-10, 2)
    Vx: Pair = (0, 20)
    Delta: Pair = (-pi / 3, pi / 3)
    S: Pair = (0, 13)


state_constraints = StateConstraints()


@unique
class IdxParams(IntEnum):
    SpeedLimit = 0
    """max longitudinal speed in allowed [m/s]"""
    TargetSpeed = 1
    """target longitudinal speed in [m/s] """
    OptCost1 = 2
    """Optimal cost achieved by the first lexicographic optimization"""
    OptCost2 = 3
    """Optimal cost achieved by the second lexicographic optimization"""
    Xobstacle = 4
    """X position of the obstacle"""
    Yobstacle = 5
    """Y position of the obstacle"""
    TargetProg = 6
    """Progress goal for the agent after which the game can be considered terminated"""
    kAboveTargetSpeedCost = 7
    """weight penalizing speed Above reference"""
    kBelowTargetSpeedCost = 8
    """weight penalizing speed Below reference"""
    kAboveSpeedLimit = 9
    """weight penalizing speed above SpeedLimit"""
    kLag = 10
    """weight penalizing lag error"""
    kLat = 11
    """weight penalizing lat error w.r.t center of the lane"""
    pLeftLane = 12
    """weight penalizing going towards the other lane"""
    kReg_dAb = 13
    """weight penalizing variations in accelerations"""
    kReg_dDelta = 14
    """weight penalizing variations in steering angle"""
    kSlack = 15
    """weight penalizing going out of the track"""
    minSafetyDistance = 16
    """Minimum safety distance"""
    carLength = 17
    """carLength"""


@dataclass(frozen=True)
class VarDesc:
    title: str
    units: str


var_descriptions = {  # todo check units
    IdxInput.dAcc: VarDesc('Change of acc', 'm/s^3'),
    IdxInput.dDelta: VarDesc('Steering rate', 'rad/s'),
    IdxInput.dS: VarDesc('Progress derivative', '-'),
    IdxInput.Slack_Lat: VarDesc('SlackLat', '-'),
    IdxInput.Slack_Coll: VarDesc('SlackColl', '-'),
    IdxInput.Slack_Obs: VarDesc('SlackObs', '-'),
    IdxState.X: VarDesc('x-position', 'm'),
    IdxState.Y: VarDesc('y-position', 'm'),
    IdxState.Theta: VarDesc('Orientation', 'rad'),
    IdxState.Vx: VarDesc('Forward velocity', 'm/s'),
    IdxState.Acc: VarDesc('Combined acceleration', 'm/s^2'),
    IdxState.Delta: VarDesc('Steering angle', 'rad'),
    IdxState.S: VarDesc('Progress', '-'),
    IdxState.CumSlackCost: VarDesc('Cumulative Slack', '-'),
    IdxState.CumLatSpeedCost: VarDesc('Cumulative Rules', '-'),
}
