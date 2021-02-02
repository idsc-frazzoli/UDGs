from dataclasses import dataclass
from enum import IntEnum
from numbers import Real
from .indices import IdxState, IdxInput, IdxParams


@dataclass(frozen=True)
class HumanConstraintsParams:
    N: int = 31
    """The mpc horizon """
    s_idx: IntEnum = IdxState
    i_idx: IntEnum = IdxInput
    p_idx: IntEnum = IdxParams
    n_states: int = len([s.value for s in IdxState])
    n_inputs: int = len([u.value for u in IdxInput])
    n_var: int = n_states + n_inputs
    n_param: int = len([u.value for u in IdxParams])
    n_bspline_points: int = 15
    dt_integrator_step: float = 0.1
