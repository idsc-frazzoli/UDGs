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
    # tv = 3
    """torque vectoring"""
    slack = 3
    """TODO"""


@unique
class IdxState(IntEnum):
    x = 4
    """x-position in world frame"""
    y = 5
    """y-position in world frame"""
    theta = 6
    """orientation in world frame"""
    dtheta = 7
    """angular velocity around z-axis"""
    vx = 8
    """forward velocity in kart frame"""
    # vy = 10
    """lateral velocity in kart frame"""
    ab = 9
    """acceleration and braking"""
    beta = 10
    """steering angle?"""
    s = 11
    """path progress in local spline reference system"""


@unique
class IdxParams(IntEnum):
    maxspeed = 0
    targetspeed = 1
    optcost1 = 2
    optcost2 = 3
    Xobstacle = 4
    Yobstacle = 5
    targetprog = 6
    pspeedcostA = 7
    pspeedcostB = 8
    pspeedcostM = 9
    plag = 10
    plat = 11
    pLeftLane = 12
    pab = 13
    pdotbeta = 14
    pslack = 15
    distance = 16
    carLength = 17

@dataclass(frozen=True)
class VarDesc:
    title: str
    units: str


var_descriptions = {  # todo check units
    IdxInput.dAb: VarDesc('Change of acc', 'm/s^3'),
    IdxInput.dBeta: VarDesc('Steering rate', 'rad/s'),
    IdxInput.ds: VarDesc('Progress derivative', '-'),
#    IdxInput.tv: VarDesc('Torque vectoring', 'Nm'),
    IdxInput.slack: VarDesc('Slack', '-'),
    IdxState.x: VarDesc('x-position', 'm'),
    IdxState.y: VarDesc('y-position', 'm'),
    IdxState.theta: VarDesc('Orientation', 'rad'),
    IdxState.dtheta: VarDesc('Angular velocity around z', 'rad/s'),
    IdxState.vx: VarDesc('Forward velocity', 'm/s'),
   # IdxState.vy: VarDesc('Lateral velocity', 'm/s'),
    IdxState.ab: VarDesc('Combined acceleration', 'm/s^2'),
    IdxState.beta: VarDesc('Steering angle', 'rad'),
    IdxState.s: VarDesc('Progress', '-'),
}
