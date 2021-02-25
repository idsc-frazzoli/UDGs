from typing import NewType

from udgs_models.model_def.parameters import CarParams

params = CarParams()
x_idx = params.x_idx
u_idx = params.u_idx
p_idx = params.p_idx

SolutionMethod = NewType("SolutionMethod", str)
PG = SolutionMethod("PG")
LexicographicPG = SolutionMethod("LexicographicPG")
IBR = SolutionMethod("IBR")
LexicographicIBR = SolutionMethod("LexicographicIBR")
AVAILABLE_METHODS = (PG, LexicographicPG, IBR, LexicographicIBR)

ForcesModel = NewType("ForcesModel", str)
ForcesPG = ForcesModel("ForcesPG")
ForcesIBR = ForcesModel("ForcesIBR")
