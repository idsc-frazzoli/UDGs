from collections import namedtuple
from dataclasses import dataclass
from enum import IntEnum, unique

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
    dAb = 0
    """desired change of acceleration"""
    dBeta = 1
    """rate of change steering angle"""
    ds = 2
    """Derivative of the progress along the track"""
    slack = 3
    """Slack variable for violating constraints"""

@unique
class IdxState(IntEnum):
    x = 4
    """x-position in world frame"""
    y = 5
    """y-position in world frame"""
    theta = 6
    """orientation in world frame"""
    v = 7
    """vehicle speed in car frame"""
    ab = 8
    """acceleration and braking"""
    beta = 9
    """steering angle"""
    s = 10
    """path progress in local spline reference system"""
    latSpeedCost = 11
    """left-lane lateral error + exceeding maximum speed cost integral state"""
    collCost = 12
    """collision cost integral state"""

@unique
class IdxParams(IntEnum):
    targetSpeed = 0
    """target speed for the vehicle"""
    maxSpeed = 1
    """road speed limit for the vehicle"""
    firstOptCost = 2
    """total collision cost from first optimization problem"""
    secondOptCost = 3
    """total lateral error cost + speed cost for exceeding speed limits from second optimization problem"""
    progressMax = 4
    """progress required to pass the intersection"""
    obstaclePosX = 5
    obstaclePosY = 6
    """X and Y position Coordinates of an obstacle"""
    pLag = 7
    pLat = 8
    """Lateral and Lag weights, considered w.r.t centerline"""
    pLeftLane = 9
    """Left lane penalization term"""
    pSpeedCostB = 10
    pSpeedCostA = 11
    pSpeedCostM = 12
    """Speed weights for costs : 1) B Below reference 2) A Above reference 3) M Above Max"""
    pab = 13
    pDotBeta = 14
    """Regularizing Terms"""
    pSlack = 15
    """Collision weight"""
    distance = 16
    """maximum distance allowed between agents"""
    carLength = 17
    """car length parameter"""

@dataclass(frozen=True)
class VarDesc:
    title: str
    units: str


var_descriptions = {  # todo check units
    IdxInput.dAb: VarDesc('Change of acc', 'm/s^3'),
    IdxInput.dBeta: VarDesc('Steering rate', 'rad/s'),
    IdxInput.ds: VarDesc('Progress derivative', '-'),
    # IdxInput.tv: VarDesc('Torque vectoring', 'Nm'),
    IdxInput.slack: VarDesc('Slack', '-'),
    IdxState.x: VarDesc('x-position', 'm'),
    IdxState.y: VarDesc('y-position', 'm'),
    IdxState.theta: VarDesc('Orientation', 'rad'),
    # IdxState.dtheta: VarDesc('Angular velocity around z', 'rad/s'),
    IdxState.v: VarDesc('Forward velocity', 'm/s'),
    # IdxState.vy: VarDesc('Lateral velocity', 'm/s'),
    IdxState.ab: VarDesc('Combined acceleration', 'm/s^2'),
    IdxState.beta: VarDesc('Steering angle', 'rad'),
    IdxState.s: VarDesc('Progress', '-'),
    IdxState.latSpeedCost: VarDesc('left-lane lateral error + exceeding maximum speed integral term', '-'),
    IdxState.collCost: VarDesc('collision cost integral term', '-'),
}
