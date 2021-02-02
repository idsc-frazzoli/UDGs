% MPCCgokart - a fast solver generated by FORCES PRO v4.1.1
%
%   OUTPUT = MPCCgokart(PARAMS) solves a multistage problem
%   subject to the parameters supplied in the following struct:
%       PARAMS.xinit - column vector of length 8
%       PARAMS.x0 - column vector of length 372
%       PARAMS.all_parameters - column vector of length 2015
%
%   OUTPUT returns the values of the last iteration of the solver where
%       OUTPUT.all_var - column vector of size 372
%
%   [OUTPUT, EXITFLAG] = MPCCgokart(PARAMS) returns additionally
%   the integer EXITFLAG indicating the state of the solution with 
%       1 - OPTIMAL solution has been found (subject to desired accuracy)
%       0 - Timeout - maximum number of iterations reached
%      -6 - NaN or INF occured during evaluation of functions and derivatives. Please check your initial guess.
%      -7 - Method could not progress. Problem may be infeasible. Run FORCESdiagnostics on your problem to check for most common errors in the formulation.
%    -100 - License error
%
%   [OUTPUT, EXITFLAG, INFO] = MPCCgokart(PARAMS) returns 
%   additional information about the last iterate:
%       INFO.it        - number of iterations that lead to this result
%       INFO.res_eq    - max. equality constraint residual
%       INFO.res_ineq  - max. inequality constraint residual
%       INFO.rsnorm    - norm of stationarity condition
%       INFO.rcompnorm    - max of all complementarity violations
%       INFO.pobj      - primal objective
%       INFO.mu        - duality measure
%       INFO.solvetime - Time needed for solve (wall clock time)
%       INFO.fevalstime - Time needed for function evaluations (wall clock time)
%
% See also COPYING
