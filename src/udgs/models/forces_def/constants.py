from typing import NewType

__all__ = ["SolutionMethod",
           "PG",
           "LexicographicPG",
           "IBR",
           "LexicographicIBR",
           "AVAILABLE_METHODS",
           "ForcesModel",
           "ForcesPG",
           "ForcesIBR",
           ]

SolutionMethod = NewType("SolutionMethod", str)
PG = SolutionMethod("PG")
LexicographicPG = SolutionMethod("LexicographicPG")
IBR = SolutionMethod("IBR")
LexicographicIBR = SolutionMethod("LexicographicIBR")
AVAILABLE_METHODS = (PG, LexicographicPG, IBR, LexicographicIBR)

ForcesModel = NewType("ForcesModel", str)
ForcesPG = ForcesModel("ForcesPG")
ForcesIBR = ForcesModel("ForcesIBR")
