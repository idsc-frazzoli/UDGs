from dataclasses import dataclass
from typing import Mapping
import numpy as np
from frozendict import frozendict

__all__ = ["DriverConfig", "BehaviorSpec", "behaviors_zoo"]


@dataclass(frozen=True, unsafe_hash=True)
class DriverConfig:
    maxspeed: float
    """max longitudinal speed in [m/s]"""
    targetspeed: float
    """target longitudinal speed in [m/s]"""
    optcost1: float
    """optimal cost of first optimization"""
    optcost2: float
    """optimal cost of second optimization"""
    Xobstacle: float
    """X position of the obstacle"""
    Yobstacle: float
    """Y position of the obstacle"""
    targetprog: float
    """target progress at the end of the prediction horizon"""
    pspeedcostA: float
    """weight penalizing speed Above reference"""
    pspeedcostB: float
    """weight penalizing speed Below reference"""
    pspeedcostM: float
    """weight penalizing speed above SpeedLimit"""
    plag: float
    """weight penalizing lag error"""
    plat: float
    """weight penalizing lat error w.r.t center of the lane"""
    pLeftLane: float
    """weight penalizing going towards the other lane"""
    pab: float
    """weight penalizing variations in accelerations"""
    pdotbeta: float
    """weight penalizing variations in steering angle"""
    pslack: float
    """weight penalizing going out of the track"""
    distance: float
    """max distance allowed"""
    carLength: float
    """carLength"""

@dataclass(frozen=True, unsafe_hash=True)
class BehaviorSpec:
    desc: str
    """Description of the behavior"""
    config: DriverConfig


_cautious = DriverConfig(
    maxspeed=9,
    targetspeed=8.3,
    optcost1=100000,
    optcost2=100000,
    Xobstacle=50,
    Yobstacle=37,
    targetprog=7,
    pspeedcostA=2,
    pspeedcostB=0.5,
    pspeedcostM=4,
    plag=1,
    plat=1,
    pLeftLane=1,
    pab=0.0006,
    pdotbeta=2,
    pslack=10000,
    distance=3,
    carLength=2.5,
)
cautious_spec = BehaviorSpec(desc="Config1", config=_cautious)

_initializationConfig = DriverConfig(
    maxspeed=9,
    targetspeed=8.3,
    optcost1=10000000,
    optcost2=10000000,
    Xobstacle=50,
    Yobstacle=37,
    targetprog=7,
    pspeedcostA=2,
    pspeedcostB=0.1,
    pspeedcostM=4,
    plag=1,
    plat=1,
    pLeftLane=1,
    pab=0.006,
    pdotbeta=2,
    pslack=1000000,
    distance=0,
    carLength=2.5,
)
initialization_spec = BehaviorSpec(desc="initConfig", config=_initializationConfig)

_firstOptim = DriverConfig(
    maxspeed=9,
    targetspeed=8.3,
    optcost1=100000,
    optcost2=100000,
    Xobstacle=50,
    Yobstacle=37,
    targetprog=7,
    pspeedcostA=0,
    pspeedcostB=0,
    pspeedcostM=0,
    plag=1,
    plat=0,
    pLeftLane=0,
    pab=0,
    pdotbeta=0,
    pslack=1000000,
    distance=3,
    carLength=2.5,
)
firstOptim_spec = BehaviorSpec(desc="firstOptim", config=_firstOptim)

_secondOptim = DriverConfig(
    maxspeed=9,
    targetspeed=8.3,
    optcost1=100000,
    optcost2=100000,
    Xobstacle=50,
    Yobstacle=37,
    targetprog=7,
    pspeedcostA=0,
    pspeedcostB=0,
    pspeedcostM=4,
    plag=1,
    plat=0,
    pLeftLane=6,
    pab=0,
    pdotbeta=0,
    pslack=1000000,
    distance=3,
    carLength=2.5,
)
secondOptim_spec = BehaviorSpec(desc="secondOptim", config=_secondOptim)

behaviors_zoo: Mapping[str, BehaviorSpec] = frozendict(
    {
        "Config1": cautious_spec,
        "initConfig": initialization_spec,
        "firstOptim": firstOptim_spec,
        "secondOptim": secondOptim_spec,
    }
)
