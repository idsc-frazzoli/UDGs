from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Dict
from frozendict import frozendict

__all__ = ["BehaviorSpec", "behaviors_zoo"]

from udgs.forces_models.model_def import IdxParams


@dataclass(frozen=True, unsafe_hash=True)
class BehaviorSpec:
    desc: str
    """Description of the behavior"""
    config: Dict


_default = dict.fromkeys(IdxParams)
_default[IdxParams.SpeedLimit] = 9
_default[IdxParams.TargetSpeed] = 7
_default[IdxParams.OptCost1] = 100000
_default[IdxParams.OptCost2] = 100000
_default[IdxParams.Xobstacle] = 50
_default[IdxParams.Yobstacle] = 37
_default[IdxParams.TargetProg] = 7  # 7
_default[IdxParams.kAboveTargetSpeedCost] = 2
_default[IdxParams.kBelowTargetSpeedCost] = 0.4
_default[IdxParams.kAboveSpeedLimit] = 4
_default[IdxParams.kLag] = 1
_default[IdxParams.kLat] = 1
_default[IdxParams.pLeftLane] = 2
_default[IdxParams.kReg_dAb] = 0.004
_default[IdxParams.kReg_dDelta] = 2
_default[IdxParams.kSlack] = 1000
_default[IdxParams.minSafetyDistance] = 3.3  # 3
_default[IdxParams.carLength] = 2.5

cautious_spec = BehaviorSpec(desc="PG", config=_default)

_initializationConfig = deepcopy(_default)
_initializationConfig[IdxParams.minSafetyDistance] = 0
initialization_spec = BehaviorSpec(desc="initConfig", config=_initializationConfig)

_firstOptim = deepcopy(_default)
_firstOptim[IdxParams.kAboveSpeedLimit] = 0
_firstOptim[IdxParams.kAboveTargetSpeedCost] = 0
_firstOptim[IdxParams.kBelowTargetSpeedCost] = 0
_firstOptim[IdxParams.kReg_dAb] = 0
_firstOptim[IdxParams.kReg_dDelta] = 0
_firstOptim[IdxParams.kLat] = 0
_firstOptim[IdxParams.pLeftLane] = 0
firstOptim_spec = BehaviorSpec(desc="firstOptim", config=_firstOptim)

_secondOptim = deepcopy(_default)
_secondOptim[IdxParams.kAboveTargetSpeedCost] = 0
_secondOptim[IdxParams.kBelowTargetSpeedCost] = 0
_secondOptim[IdxParams.pLeftLane] = 3
_secondOptim[IdxParams.kReg_dAb] = 0
_secondOptim[IdxParams.kReg_dDelta] = 0

secondOptim_spec = BehaviorSpec(desc="secondOptim", config=_secondOptim)
_thirdOptim = deepcopy(_default)
thirdOptim_spec = BehaviorSpec(desc="thirdOptim", config=_thirdOptim)

behaviors_zoo: Mapping[str, BehaviorSpec] = frozendict(
    {
        "PG": cautious_spec,
        "ibr": cautious_spec,
        "initConfig": initialization_spec,
        "firstOptim": firstOptim_spec,
        "secondOptim": secondOptim_spec,
        "thirdOptim": thirdOptim_spec,
    }
)
