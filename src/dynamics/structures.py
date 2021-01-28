from dataclasses import dataclass
import numpy as np
from geometry import SE2_from_xytheta


@dataclass
class GokartState:
    # not used for now
    x: float
    y: float
    psi: float
    vx: float
    vy: float
    dpsi: float

    def pose(self) -> np.ndarray:
        return SE2_from_xytheta(self.x, self.y, self.psi)
