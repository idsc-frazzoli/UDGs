/*
MPCCgokart : A fast customized optimization solver.

Copyright (C) 2013-2020 EMBOTECH AG [info@embotech.com]. All rights reserved.


This software is intended for simulation and testing purposes only. 
Use of this software for any commercial purpose is prohibited.

This program is distributed in the hope that it will be useful.
EMBOTECH makes NO WARRANTIES with respect to the use of the software 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. 

EMBOTECH shall not have any liability for any damage arising from the use
of the software.

This Agreement shall exclusively be governed by and interpreted in 
accordance with the laws of Switzerland, excluding its principles
of conflict of laws. The Courts of Zurich-City shall have exclusive 
jurisdiction in case of any dispute.

*/

/* Generated by FORCES PRO v4.1.1 on Tuesday, February 2, 2021 at 9:08:10 AM */

#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif

#ifndef MPCCgokart_H
#define MPCCgokart_H

#ifndef SOLVER_STANDARD_TYPES
#define SOLVER_STANDARD_TYPES

typedef signed char solver_int8_signed;
typedef unsigned char solver_int8_unsigned;
typedef char solver_int8_default;
typedef signed short int solver_int16_signed;
typedef unsigned short int solver_int16_unsigned;
typedef short int solver_int16_default;
typedef signed int solver_int32_signed;
typedef unsigned int solver_int32_unsigned;
typedef int solver_int32_default;
typedef signed long long int solver_int64_signed;
typedef unsigned long long int solver_int64_unsigned;
typedef long long int solver_int64_default;

#endif


/* DATA TYPE ------------------------------------------------------------*/
typedef double MPCCgokart_float;

typedef double MPCCgokartinterface_float;

/* SOLVER SETTINGS ------------------------------------------------------*/

/* MISRA-C compliance */
#ifndef MISRA_C_MPCCgokart
#define MISRA_C_MPCCgokart (0)
#endif

/* restrict code */
#ifndef RESTRICT_CODE_MPCCgokart
#define RESTRICT_CODE_MPCCgokart (0)
#endif

/* print level */
#ifndef SET_PRINTLEVEL_MPCCgokart
#define SET_PRINTLEVEL_MPCCgokart    (0)
#endif

/* timing */
#ifndef SET_TIMING_MPCCgokart
#define SET_TIMING_MPCCgokart    (1)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define SET_MAXIT_MPCCgokart			(200)	

/* scaling factor of line search (FTB rule) */
#define SET_FLS_SCALE_MPCCgokart		(MPCCgokart_float)(0.99)      

/* maximum number of supported elements in the filter */
#define MAX_FILTER_SIZE_MPCCgokart	(200) 

/* maximum number of supported elements in the filter */
#define MAX_SOC_IT_MPCCgokart			(4) 

/* desired relative duality gap */
#define SET_ACC_RDGAP_MPCCgokart		(MPCCgokart_float)(0.0001)

/* desired maximum residual on equality constraints */
#define SET_ACC_RESEQ_MPCCgokart		(MPCCgokart_float)(1E-06)

/* desired maximum residual on inequality constraints */
#define SET_ACC_RESINEQ_MPCCgokart	(MPCCgokart_float)(1E-06)

/* desired maximum violation of complementarity */
#define SET_ACC_KKTCOMPL_MPCCgokart	(MPCCgokart_float)(1E-06)


/* SOLVER RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define OPTIMAL_MPCCgokart      (1)

/* maximum number of iterations has been reached */
#define MAXITREACHED_MPCCgokart (0)

/* solver has stopped due to a timeout */
#define TIMEOUT_MPCCgokart   (2)

/* wrong number of inequalities error */
#define INVALID_NUM_INEQ_ERROR_MPCCgokart  (-4)

/* factorization error */
#define FACTORIZATION_ERROR_MPCCgokart   (-5)

/* NaN encountered in function evaluations */
#define BADFUNCEVAL_MPCCgokart  (-6)

/* no progress in method possible */
#define NOPROGRESS_MPCCgokart   (-7)

/* invalid values in parameters */
#define PARAM_VALUE_ERROR_MPCCgokart   (-11)

/* too small timeout given */
#define INVALID_TIMEOUT_MPCCgokart   (-12)

/* licensing error - solver not valid on this machine */
#define LICENSE_ERROR_MPCCgokart  (-100)

/* INTEGRATORS RETURN CODE ------------*/
/* Integrator ran successfully */
#define INTEGRATOR_SUCCESS (11)
/* Number of steps set by user exceeds maximum number of steps allowed */
#define INTEGRATOR_MAXSTEPS_EXCEEDED (12)



/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct
{
    /* vector of size 8 */
    MPCCgokart_float xinit[8];

    /* vector of size 372 */
    MPCCgokart_float x0[372];

    /* vector of size 2015 */
    MPCCgokart_float all_parameters[2015];


} MPCCgokart_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct
{
    /* vector of size 372 */
    MPCCgokart_float all_var[372];


} MPCCgokart_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct
{
    /* iteration number */
    solver_int32_default it;

	/* number of iterations needed to optimality (branch-and-bound) */
	solver_int32_default it2opt;
	
    /* inf-norm of equality constraint residuals */
    MPCCgokart_float res_eq;
	
    /* inf-norm of inequality constraint residuals */
    MPCCgokart_float res_ineq;

	/* norm of stationarity condition */
    MPCCgokart_float rsnorm;

	/* max of all complementarity violations */
    MPCCgokart_float rcompnorm;

    /* primal objective */
    MPCCgokart_float pobj;	
	
    /* dual objective */
    MPCCgokart_float dobj;	

    /* duality gap := pobj - dobj */
    MPCCgokart_float dgap;		
	
    /* relative duality gap := |dgap / pobj | */
    MPCCgokart_float rdgap;		

    /* duality measure */
    MPCCgokart_float mu;

	/* duality measure (after affine step) */
    MPCCgokart_float mu_aff;
	
    /* centering parameter */
    MPCCgokart_float sigma;
	
    /* number of backtracking line search steps (affine direction) */
    solver_int32_default lsit_aff;
    
    /* number of backtracking line search steps (combined direction) */
    solver_int32_default lsit_cc;
    
    /* step size (affine direction) */
    MPCCgokart_float step_aff;
    
    /* step size (combined direction) */
    MPCCgokart_float step_cc;    

	/* solvertime */
	MPCCgokart_float solvetime;   

	/* time spent in function evaluations */
	MPCCgokart_float fevalstime;  


} MPCCgokart_info;







/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* Time of Solver Generation: (UTC) Tuesday, February 2, 2021 9:08:12 AM */
/* User License expires on: (UTC) Monday, February 15, 2021 10:00:00 PM (approx.) (at the time of code generation) */
/* Solver Static License expires on: (UTC) Monday, February 15, 2021 10:00:00 PM (approx.) */
/* Solver Generation Request Id: c0532912-4891-44c8-a718-486cf03791b0 */
/* examine exitflag before using the result! */
#ifdef __cplusplus
extern "C" {
#endif		

typedef void (*MPCCgokart_extfunc)(MPCCgokart_float* x, MPCCgokart_float* y, MPCCgokart_float* lambda, MPCCgokart_float* params, MPCCgokart_float* pobj, MPCCgokart_float* g, MPCCgokart_float* c, MPCCgokart_float* Jeq, MPCCgokart_float* h, MPCCgokart_float* Jineq, MPCCgokart_float* H, solver_int32_default stage, solver_int32_default iterations, solver_int32_default threadID);

extern solver_int32_default MPCCgokart_solve(MPCCgokart_params *params, MPCCgokart_output *output, MPCCgokart_info *info, FILE *fs, MPCCgokart_extfunc evalextfunctions_MPCCgokart);	







#ifdef __cplusplus
}
#endif

#endif
