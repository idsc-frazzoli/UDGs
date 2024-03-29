from dataclasses import dataclass
from enum import IntEnum
from .indices import IdxState, IdxInput, IdxParams


@dataclass(frozen=True)
class CarParams:
    N: int = 60
    """ # of steps in the prediction horizon """
    dt_integrator_step: float = 0.1
    """ step length in seconds """
    x_idx: IntEnum = IdxState
    u_idx: IntEnum = IdxInput
    p_idx: IntEnum = IdxParams
    n_states: int = len([s.value for s in IdxState])
    n_inputs: int = len([u.value for u in IdxInput])
    n_var: int = n_states + n_inputs
    n_opt_param: int = len([u.value for u in IdxParams])
    n_bspline_points: int = 18
    n_all_param: int = n_opt_param + n_bspline_points * 3

params = CarParams()
x_idx = params.x_idx
u_idx = params.u_idx
p_idx = params.p_idx
