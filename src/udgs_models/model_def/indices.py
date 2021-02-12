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
    Slack = 3
    """TODO"""


Pair = Tuple[float, float]


@dataclass(frozen=True)
class InputConstraints:
    dS: Pair = (-1, 5)
    dAcc: Pair = (-2, 2)


input_constraints = InputConstraints()


@unique
class IdxState(IntEnum):
    X = 4
    """x-position in world frame"""
    Y = 5
    """y-position in world frame"""
    Theta = 6
    """orientation in world frame"""
    Vx = 7
    """forward velocity in kart frame"""
    Acc = 8
    """acceleration and braking"""
    Delta = 9
    """steering angle"""
    S = 10
    """path progress in local spline reference system"""
    CumSlackCost = 11
    """cumulative slack"""
    CumLatSpeedCost = 12
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
    TargetSpeed = 1
    OptCost1 = 2
    """Optimal cost achieved by the first lexicographic optimization"""
    OptCost2 = 3
    """Optimal cost achieved by the second lexicographic optimization"""
    Xobstacle = 4
    Yobstacle = 5
    TargetProg = 6
    """Progress goal for the agent after which the game can be considered terminated"""
    kAboveTargetSpeedCost = 7
    kBelowTargetSpeedCost = 8
    kAboveSpeedLimit = 9
    kLag = 10
    kLat = 11
    pLeftLane = 12
    kReg_dAb = 13
    kReg_dDelta = 14
    kSlack = 15
    minSafetyDistance = 16
    """Minimum safety distance"""
    carLength = 17


@dataclass(frozen=True)
class VarDesc:
    title: str
    units: str


var_descriptions = {  # todo check units
    IdxInput.dAcc: VarDesc('Change of acc', 'm/s^3'),
    IdxInput.dDelta: VarDesc('Steering rate', 'rad/s'),
    IdxInput.dS: VarDesc('Progress derivative', '-'),
    IdxInput.Slack: VarDesc('Slack', '-'),
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
