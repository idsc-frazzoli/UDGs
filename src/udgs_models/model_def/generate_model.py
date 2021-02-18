from udgs_models.model_def.dynamics_car import _dynamics_car, dynamics_cars

import numpy as np

from . import *
from .indices import input_constraints, state_constraints
from .objective import objective_car
from .nlconstraints import nlconst_car, nlconst_carN, nlconst_car_ibr, nlconst_car_ibrN
from forcespro import nlp, CodeOptions

__all__ = ["generate_car_model"]


def generate_car_model(generate_solver: bool, to_deploy: bool, num_cars: int, condition: int):
    """
    This model assumes:
        - a state given by ...
        - control inputs given by ...

    :return:
    """
    if condition == 0 or condition == 1:  # PG or Lexicographic PG
        solver_name: str = "Forces_udgs_solver"
        model = nlp.SymbolicModel(params.N)
        model.nvar = params.n_var * num_cars
        model.neq = params.n_states * num_cars

        # Number of parameters
        model.npar = params.n_opt_param + 3 * num_cars * params.n_bspline_points
        # indices of the left hand side of the dynamical constraint
        model.E = np.concatenate(
            [np.zeros((num_cars * params.n_states, num_cars * params.n_inputs)), np.eye(num_cars * params.n_states)],
            axis=1)
        model.continuous_dynamics = dynamics_cars[num_cars]

        collision_constraints = int(num_cars * (num_cars - 1) / 2)
        obstacle_constraints = num_cars
        # inequality constraints
        model.nh = 2 * num_cars + collision_constraints + obstacle_constraints  # Number of inequality constraints
        model.ineq = nlconst_car[num_cars]
        model.hu = []
        model.hl = []
        for k in range(model.nh):
            model.hu = np.append(model.hu, np.array(0))  # upper bound for nonlinear constraints
            model.hl = np.append(model.hl, np.array(-np.inf))  # lower bound for nonlinear constraints

        # Terminal State Constraints
        model.nhN = 3 * num_cars + collision_constraints + obstacle_constraints + 2  # Number of inequality constraints
        model.ineqN = nlconst_carN[num_cars]
        model.huN = []
        model.hlN = []
        for k in range(model.nhN):
            model.huN = np.append(model.huN, np.array(0))  # upper bound for nonlinear constraints
            model.hlN = np.append(model.hlN, np.array(-np.inf))  # lower bound for nonlinear constraints

        for i in range(params.N):
            model.objective[i] = objective_car[num_cars]

        model.xinitidx = range(params.n_inputs * num_cars, params.n_var * num_cars)

        # Equality constraints
        model.ub = np.ones(params.n_var * num_cars) * np.inf
        model.lb = -np.ones(params.n_var * num_cars) * np.inf

        for k in range(num_cars):
            # delta path progress
            upd_s_idx = k * params.n_states + (num_cars - 1) * params.n_inputs
            upd_i_idx = k * params.n_inputs

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
    else:  # IBR or Lexicographic IBR
        solver_name: str = "Forces_udgs_solver_IBR"
        model = nlp.SymbolicModel(params.N)
        model.nvar = params.n_var
        model.neq = params.n_states

        # Number of parameters
        model.npar = params.n_opt_param + 3 * params.n_bspline_points
        # indices of the left hand side of the dynamical constraint
        model.E = np.concatenate([np.zeros((params.n_states, params.n_inputs)), np.eye(params.n_states)], axis=1)
        model.continuous_dynamics = dynamics_cars[1]

        collision_constraints = num_cars - 1
        obstacle_constraints = 1
        # inequality constraints
        model.nh = 2 + collision_constraints + obstacle_constraints  # Number of inequality constraints
        model.ineq = nlconst_car_ibr[num_cars]
        model.hu = []
        model.hl = []
        for k in range(model.nh):
            model.hu = np.append(model.hu, np.array(0))  # upper bound for nonlinear constraints
            model.hl = np.append(model.hl, np.array(-np.inf))  # lower bound for nonlinear constraints

        # Terminal State Constraints
        model.nhN = 5 + collision_constraints + obstacle_constraints  # Number of inequality constraints
        model.ineqN = nlconst_car_ibrN[num_cars]
        model.huN = []
        model.hlN = []
        for k in range(model.nhN):
            model.huN = np.append(model.huN, np.array(0))  # upper bound for nonlinear constraints
            model.hlN = np.append(model.hlN, np.array(-np.inf))  # lower bound for nonlinear constraints

        for i in range(params.N):
            model.objective[i] = objective_car[1]

        model.xinitidx = range(params.n_inputs, params.n_var)

        # Equality constraints
        model.ub = np.ones(params.n_var) * np.inf
        model.lb = -np.ones(params.n_var) * np.inf

        # delta path progress
        model.lb[params.u_idx.dS] = input_constraints.dS[0]
        model.ub[params.u_idx.dS] = input_constraints.dS[1]

        # Forward force lower bound
        model.lb[params.u_idx.dAcc] = input_constraints.dAcc[0]
        model.ub[params.u_idx.dAcc] = input_constraints.dAcc[1]

        # slack limit
        model.lb[params.u_idx.Slack_Lat] = 0
        model.lb[params.u_idx.Slack_Coll] = 0
        model.lb[params.u_idx.Slack_Obs] = 0

        # Forward force lower bound
        model.lb[params.x_idx.Acc] = state_constraints.Acc[0]
        model.ub[params.x_idx.Acc] = state_constraints.Acc[1]

        # Speed lower bound
        model.lb[params.x_idx.Vx] = state_constraints.Vx[0]
        model.ub[params.x_idx.Vx] = state_constraints.Vx[1]

        # Steering Angle Bounds
        model.lb[params.x_idx.Delta] = state_constraints.Delta[0]
        model.ub[params.x_idx.Delta] = state_constraints.Delta[1]

        # Path  Progress  Bounds
        model.lb[params.x_idx.S] = state_constraints.S[0]
        model.ub[params.x_idx.S] = state_constraints.S[1]

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
        if condition == 0 or condition == 1:  # PG or Lexicographic PG
            output_all = ("all_var", list(range(0, params.N)), list(range(0, params.n_var * num_cars)))
        else:  # IBR or Lexicographic IBR
            output_all = ("all_var", list(range(0, params.N)), list(range(0, params.n_var)))

        solver = model.generate_solver(codeoptions, [output_all])
    else:
        solver = nlp.Solver.from_directory(solver_name)
    return model, solver
