from Car_mpcc.Car_optimizer.dynamics_car import dynamics_car

import numpy as np

from . import params
from .objective import objective_car
from .nlconstraints import nlconst_car, nlconst_carN
from forcespro import nlp, CodeOptions

__all__=["generate_car_model"]

def generate_car_model(generate_solver: bool, to_deploy: bool):
    """
    This model assumes:
        - a state given by ...
        - control inputs given by ...

    :return:
    """
    solver_name: str = "MPCC_Car"
    model = nlp.SymbolicModel(params.N)
    model.nvar = params.n_var
    model.neq = params.n_states

    # Number of parameters
    model.npar = params.n_param + 3 * params.n_bspline_points
    model.eq = lambda z, p: nlp.integrate(
        dynamics_car,
        z[params.n_inputs: params.n_var],
        z[0: params.n_inputs],
        p,
        integrator=nlp.integrators.RK4,
        stepsize=params.dt_integrator_step,
    )  # todo dynamics integrator interstagedx_HC

    # model.continuous_dynamics = lambda x, u, p: dynamics_HC

    # indices of the left hand side of the dynamical constraint
    model.E = np.concatenate([np.zeros((params.n_states, params.n_inputs)), np.eye(params.n_states)], axis=1)

    # inequality constraints
    model.nh = 2  # Number of inequality constraints
    model.ineq = nlconst_car
    model.hu = np.array([0, 0])  # upper bound for nonlinear constraints
    model.hl = np.array([-np.inf, -np.inf])  # lower bound for nonlinear constraints

    # Terminal State Constraints
    model.nhN = 3  # Number of inequality constraints
    model.ineqN = nlconst_carN
    model.huN = np.array([0, 0, 0])  # upper bound for nonlinear constraints
    model.hlN = np.array([-np.inf, -np.inf, -np.inf])  # lower bound for nonlinear constraints

    for i in range(params.N):
        model.objective[i] = objective_car

    model.xinitidx = range(params.n_inputs, params.n_var)

    # Equality constraints
    model.ub = np.ones(params.n_var) * np.inf
    model.lb = -np.ones(params.n_var) * np.inf

    # delta path progress
    model.ub[params.i_idx.dots] = 5
    model.lb[params.i_idx.dots] = -1

    # Forward force lower bound
    model.lb[params.i_idx.dAb] = -10
    model.ub[params.i_idx.dAb] = 10

    # Forward force lower bound
    model.lb[params.s_idx.ab] = -np.inf
    model.ub[params.s_idx.ab] = 2
    # slack limit
    model.lb[params.i_idx.slack] = 0

    # Speed lower bound
    model.lb[params.s_idx.vx] = 0
    model.ub[params.s_idx.vx] = 20

    # Steering Angle Bounds
    model.ub[params.s_idx.beta] = 0.95
    model.lb[params.s_idx.beta] = -0.95

    # Path  Progress  Bounds
    model.ub[params.s_idx.s] = params.n_bspline_points - 2  # fixme why limiting path progress?
    model.lb[params.s_idx.s] = 0

    # CodeOptions  for FORCES solver
    codeoptions = CodeOptions(solver_name)
    codeoptions.maxit = 200  # Maximum number of iterations
    codeoptions.printlevel = 0  # Use printlevel = 2 to print progress (but not for timings)
    # 0: no optimization, 1: optimize for size, 2: optimize for speed, 3: optimize for size & speed
    codeoptions.optlevel = (2)
    codeoptions.printlevel = 0  # optional, on some platforms printing is not supported
    codeoptions.cleanup = 0  # to keep necessary files for target compile
    codeoptions.timing = 1
    codeoptions.overwrite = 1  # 1: overwrite existing solver
    codeoptions.BuildSimulinkBlock = 0
    codeoptions.noVariableElimination = 1
    codeoptions.nlp.checkFunctions = 0
    if to_deploy:
        codeoptions.useFloatingLicense = 1  # Comment out unless you got a floating license
        codeoptions.platform = (
            "Docker-Gnu-x86_64"  # Comment    out  unless  you  got  a  SW / testing  license
        )
    # codeoptions.nlp.integrator.type = 'RK4'
    # codeoptions.nlp.integrator.Ts = params.integrator_stepsize
    # codeoptions.nlp.integrator.nodes = 5  # fixme what is this?

    if generate_solver:
        # necessary to have all the zs stack in one vector
        output_all = ("all_var", list(range(0, params.N)), list(range(0, params.n_var)))
        solver = model.generate_solver(codeoptions, [output_all])
    else:
        solver = nlp.Solver.from_directory(solver_name)
    return model, solver
