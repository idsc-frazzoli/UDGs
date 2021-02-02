from dataclasses import dataclass
from typing import Mapping

from frozendict import frozendict

__all__ = ["DriverConfig", "BehaviorSpec", "behaviors_zoo"]


@dataclass(frozen=True, unsafe_hash=True)
class DriverConfig:
    targetSpeed: float
    """target ?longitudinal? speed in [m/s]"""
    maxSpeed: float
    """max longitudinal acc in [m/s^-1]"""
    firstOptCost: float
    """"""
    secondOptCost: float
    """"""
    progressMax: float
    """"""
    obstaclePosX: float
    """"""
    obstaclePosY: float
    """"""
    pLag: float
    """"""
    pLat: float
    """"""
    pLeftLane: float
    """"""
    pSpeedCostB: float
    """"""
    pSpeedCostA: float
    """"""
    pSpeedCostM: float
    """"""
    pab: float
    """"""
    pDotBeta: float
    """"""
    pSlack: float
    """"""
    distance: float
    """"""
    carLength: float
    """"""

@dataclass(frozen=True, unsafe_hash=True)
class BehaviorSpec:
    desc: str
    """Description of the behavior"""
    config: DriverConfig


_cautious = DriverConfig(
    targetSpeed=9,
    maxSpeed=8.3,
    firstOptCost=0,
    secondOptCost=0,
    progressMax=3,
    obstaclePosX=0,
    obstaclePosY=0,
    pLag=1,
    pLat=0,
    pLeftLane=1.2,
    pSpeedCostB=0.8,
    pSpeedCostA=3,
    pSpeedCostM=3,
    pab=0.006,
    pDotBeta=4,
    pSlack=100,
    distance=4,
    carLength=2,
)
cautious_spec = BehaviorSpec(desc="Beginner mode", config=_cautious)

# _medium = DriverConfig(
#     max_speed=7,
#     max_acc=3,
#     steering_reg=0.02,
#     specificmoi=0.3,
#     plag=1,
#     plat=0.01,
#     pprog=0.15,
#     pab=0.0006,
#     pspeedcost=0.04,
#     pslack=8,
#     ptv=0.005,
# )
# medium_spec = BehaviorSpec(desc="", config=_medium)
# _aggressive = DriverConfig(
#     max_speed=10,
#     max_acc=5,
#     steering_reg=0.02,
#     specificmoi=0.3,
#     plag=1,
#     plat=0.015,
#     pprog=0.2,
#     pab=0.0004,
#     pspeedcost=0.005,
#     pslack=7,
#     ptv=0.0075,
# )
# aggressive_spec = BehaviorSpec(
#     desc="Advanced mode enabled. Be careful, with high limits come high responsibilities", config=_aggressive
# )
# _drifting = DriverConfig(
#     max_speed=10,
#     max_acc=5,
#     steering_reg=0.02,
#     specificmoi=0.3,
#     plag=0.2,
#     plat=0.01,
#     pprog=0.1,
#     pab=0.0004,
#     pspeedcost=0.04,
#     pslack=4,
#     ptv=0.05,
# )
# drifting_spec = BehaviorSpec(desc="", config=_drifting)
#
# _custom = DriverConfig(
#     max_speed=7,
#     max_acc=5,
#     steering_reg=0.02,
#     specificmoi=0.3,
#     plag=1,
#     plat=0.0001,
#     pprog=0.2,
#     pab=0.04,
#     pspeedcost=0.0004,
#     pslack=5,
#     ptv=0.03,
# )
# custom_spec = BehaviorSpec(desc="", config=_custom)
# _collision = DriverConfig(
#     max_speed=3,
#     max_acc=5,
#     steering_reg=0.2,
#     specificmoi=0.3,
#     plag=1,
#     plat=1,
#     pprog=0.2,
#     pab=0.04,
#     pspeedcost=1,
#     pslack=5,
#     ptv=0.01,
# )
# collision_spec = BehaviorSpec(desc="todo", config=_collision)

behaviors_zoo: Mapping[str, BehaviorSpec] = frozendict(
    {
        "beginner": cautious_spec,
        # "medium": medium_spec,
        # "advanced": aggressive_spec,
        # "drifting": drifting_spec,
        # "custom": custom_spec,
        # "collision": collision_spec,
    }
)
