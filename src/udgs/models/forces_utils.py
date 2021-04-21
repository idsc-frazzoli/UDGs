from typing import Optional

__all__ = ["ForcesException"]


class ForcesException(Exception):
    def __init__(self, exitflag: int, msg: Optional[str] = None):
        super(ForcesException, self).__init__(self.error_type(exitflag))

    @staticmethod
    def error_type(exitflag: int) -> str:
        if exitflag == 0:
            return (
                "Maximum number of iterations reached."
                "You can examine the value of optimality conditions "
                "returned inside the info struct by FORCESPRO to decide whether the point returned is acceptable."
            )
        elif exitflag == 1:
            return (
                "Local  optimal  solution  found (i.e.the  point  satisfies  the  KKT  optimality  conditions  to  "
                "the  requested  accuracy). "
            )
        elif exitflag == -4:
            return "Wrong  number  of  inequalities  input  to  solver."
        elif exitflag == -5:
            return "Error occured during matrix factorization."

        elif exitflag == -6:
            return "NaN or INF occurred during functions evaluations."

        elif exitflag == -7:
            return (
                "The solver could not proceed.Most likely cause is that the problem is infeasible. "
                "Try formulating a problem with slack variables (soft constraints) to avoid this error. "
            )
        elif exitflag == -8:
            return (
                "The internal QP solver could not proceed.This exitflag can "
                "only occur when using the SEQUENTIAL QUADRATIC "
                "PROGRAMMING ALGORITHM.The most likely cause is that an "
                "infeasible QP or a numerical unstable QP was "
                "encountered.Try increasing the hessian regularization "
                "parameter reg_hessian if this exitflag is encountered(see SQP SPECIFIC CODEOPTIONS). "
            )
        elif exitflag == -10:
            return (
                "NaN or INF occured during evaluation of functions and "
                "derivatives.If this occurs at iteration zero, try changing "
                "the initial point.For example, for a cost function 1 / 𝑥‾‾√ with an initialization 𝑥0=0, "
                "this error would occur. "
            )
        elif exitflag == -11:
            return "Invalid values in problem parameters."
        elif exitflag == -100:
            return (
                "License error.This typically happens if you are trying to "
                "execute code that has been generated with a Simulation "
                "license of FORCESPRO on another machine.Regenerate the solver using your machine. "
            )
        else:
            return f"Unknown exitflag: {exitflag}"
