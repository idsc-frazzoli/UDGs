from typing import Tuple, MutableMapping

from udgs_models.model_def.dynamics_car import dynamics_cars

import numpy as np

from . import *
from .indices import input_constraints, state_constraints
from .objective import objective_car
from .nlconstraints import nlconst_car, nlconst_carN, nlconst_car_ibr, nlconst_car_ibrN
from forcespro import nlp, CodeOptions

__all__ = ["generate_forces_models"]


def generate_forces_models(generate_solver: bool, to_deploy: bool, n_players: int):
    """
    It defines and creates all the solver for the solution methods,
    it returns a dictionary with key SolutionType
    :param generate_solver:
    :param to_deploy:
    :param n_players:
    :return:
    """
    forces_models: MutableMapping[SolutionMethod, Tuple] = {}
    methods2models = {(PG, LexicographicPG): ForcesPG,
                      (IBR, LexicographicIBR): ForcesIBR}
    for keys, forces_model in methods2models.items():
        model, solver = _generate_forces_model(generate_solver, to_deploy, n_players, forces_model)
        for method in keys:
            forces_models[method] = (model, solver)
    return forces_models


def _generate_forces_model(generate_solver: bool, to_deploy: bool, n_players: int, forces_model: ForcesModel):
    """
    This model assumes:
        - a state given by ...
        - control inputs given by ...

    :return:
    """
    solver_name = "solver_" + forces_model
    n_players_model = n_players if forces_model == ForcesPG else 1

    model = nlp.SymbolicModel(params.N)
    model.nvar = params.n_var * n_players_model
    model.neq = params.n_states * n_players_model

    if forces_model == ForcesPG:  # PG or Lexicographic PG
        # Number of parameters
        model.npar = params.n_opt_param + 3 * n_players_model * params.n_bspline_points
        collision_constraints = int(n_players_model * (n_players_model - 1) / 2)
    else:  # IBR
        # Number of parameters
        model.npar = params.n_opt_param + 3 * params.n_bspline_points + 2 * (n_players - 1)
        collision_constraints = n_players - 1

    obstacle_constraints = n_players_model
    # indices of the left hand side of the dynamical constraint
    model.E = np.concatenate([np.zeros((n_players_model * params.n_states, n_players_model * params.n_inputs)),
                              np.eye(n_players_model * params.n_states)], axis=1)
    model.continuous_dynamics = dynamics_cars[n_players_model]

    # inequality constraints
    model.nh = 2 * n_players_model + obstacle_constraints + collision_constraints  # Number of inequality constraints

    if forces_model == ForcesPG:  # PG or Lexicographic PG
        model.ineq = nlconst_car[n_players]
    else:  # IBR
        model.ineq = nlconst_car_ibr[n_players]

    model.hu = []
    model.hl = []
    for k in range(model.nh):
        model.hu = np.append(model.hu, np.array(0))  # upper bound for nonlinear constraints
        model.hl = np.append(model.hl, np.array(-np.inf))  # lower bound for nonlinear constraints

    # Terminal State Constraints
    model.nhN = 3 * n_players_model + 2 + collision_constraints + obstacle_constraints
    if forces_model == ForcesPG:  # PG or Lexicographic PG
        model.ineqN = nlconst_carN[n_players]
    else:  # IBR
        model.ineqN = nlconst_car_ibrN[n_players]

    model.huN = []
    model.hlN = []

    for k in range(model.nhN):
        model.huN = np.append(model.huN, np.array(0))  # upper bound for nonlinear constraints
        model.hlN = np.append(model.hlN, np.array(-np.inf))  # lower bound for nonlinear constraints

    for k in range(params.N):
        model.objective[k] = objective_car[n_players_model]

    model.xinitidx = range(params.n_inputs * n_players_model, params.n_var * n_players_model)

    # Equality constraints
    model.ub = np.ones(params.n_var * n_players_model) * np.inf
    model.lb = -np.ones(params.n_var * n_players_model) * np.inf

    for i in range(n_players_model):
        # delta path progress
        upd_s_idx = i * params.n_states + (n_players_model - 1) * params.n_inputs
        upd_i_idx = i * params.n_inputs

        model.lb[params.u_idx.dS + upd_i_idx] = input_constraints.dS[0]
        model.ub[params.u_idx.dS + upd_i_idx] = input_constraints.dS[1]

        # Forward force lower bound
        model.lb[params.u_idx.dAcc + upd_i_idx] = input_constraints.dAcc[0]
        model.ub[params.u_idx.dAcc + upd_i_idx] = input_constraints.dAcc[1]

        # slack limit
        model.lb[params.u_idx.Slack_Lat + upd_i_idx] = 0
        model.lb[params.u_idx.Slack_Coll + upd_i_idx] = 0
        model.lb[params.u_idx.Slack_Obs + upd_i_idx] = 0

        # Forward force lower bound
        model.lb[params.x_idx.Acc + upd_s_idx] = state_constraints.Acc[0]
        model.ub[params.x_idx.Acc + upd_s_idx] = state_constraints.Acc[1]

        # Speed lower bound
        model.lb[params.x_idx.Vx + upd_s_idx] = state_constraints.Vx[0]
        model.ub[params.x_idx.Vx + upd_s_idx] = state_constraints.Vx[1]

        # Steering Angle Bounds
        model.lb[params.x_idx.Delta + upd_s_idx] = state_constraints.Delta[0]
        model.ub[params.x_idx.Delta + upd_s_idx] = state_constraints.Delta[1]

        # Path  Progress  Bounds
        model.lb[params.x_idx.S + upd_s_idx] = state_constraints.S[0]
        model.ub[params.x_idx.S + upd_s_idx] = state_constraints.S[1]

    # CodeOptions  for FORCES solver
    codeoptions = CodeOptions(solver_name)
    codeoptions.maxit = 5000  # Maximum number of iterations
    codeoptions.printlevel = 0  # Use printlevel = 2 to print progress (but not for timings)
    # 0: no optimization, 1: optimize for size, 2: optimize for speed, 3: optimize for size & speed
    codeoptions.optlevel = 2
    codeoptions.printlevel = 0  # optional, on some platforms printing is not supported
    codeoptions.cleanup = 1  # to keep necessary files for target compile
    codeoptions.timing = 1
    codeoptions.overwrite = 1  # 1: overwrite existing solver
    codeoptions.BuildSimulinkBlock = 0
    codeoptions.noVariableElimination = 1
    codeoptions.nlp.checkFunctions = 0
    codeoptions.nlp.integrator.type = 'ERK4'
    codeoptions.nlp.integrator.Ts = params.dt_integrator_step
    codeoptions.nlp.integrator.nodes = 1

    if to_deploy:
        codeoptions.useFloatingLicense = 1  # Comment out unless you got a floating license
        codeoptions.platform = (
            "Docker-Gnu-x86_64"  # Comment    out  unless  you  got  a  SW / testing  license
        )

    if generate_solver:
        # necessary to have all the zs stack in one vector
        output_all = ("all_var", list(range(0, params.N)), list(range(0, params.n_var * n_players_model)))
        solver = model.generate_solver(codeoptions, [output_all])

    else:
        solver = nlp.Solver.from_directory(solver_name)
    return model, solver
