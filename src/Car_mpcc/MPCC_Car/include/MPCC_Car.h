/*
MPCC_Car : A fast customized optimization solver.

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

/* Generated by FORCES PRO v4.1.1 on Thursday, February 11, 2021 at 1:07:47 PM */

#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif

#ifndef MPCC_Car_H
#define MPCC_Car_H

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
typedef double MPCC_Car_float;

typedef double MPCC_Carinterface_float;

/* SOLVER SETTINGS ------------------------------------------------------*/

/* MISRA-C compliance */
#ifndef MISRA_C_MPCC_Car
#define MISRA_C_MPCC_Car (0)
#endif

/* restrict code */
#ifndef RESTRICT_CODE_MPCC_Car
#define RESTRICT_CODE_MPCC_Car (0)
#endif

/* print level */
#ifndef SET_PRINTLEVEL_MPCC_Car
#define SET_PRINTLEVEL_MPCC_Car    (0)
#endif

/* timing */
#ifndef SET_TIMING_MPCC_Car
#define SET_TIMING_MPCC_Car    (1)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define SET_MAXIT_MPCC_Car			(200)	

/* scaling factor of line search (FTB rule) */
#define SET_FLS_SCALE_MPCC_Car		(MPCC_Car_float)(0.99)      

/* maximum number of supported elements in the filter */
#define MAX_FILTER_SIZE_MPCC_Car	(200) 

/* maximum number of supported elements in the filter */
#define MAX_SOC_IT_MPCC_Car			(4) 

/* desired relative duality gap */
#define SET_ACC_RDGAP_MPCC_Car		(MPCC_Car_float)(0.0001)

/* desired maximum residual on equality constraints */
#define SET_ACC_RESEQ_MPCC_Car		(MPCC_Car_float)(1E-06)

/* desired maximum residual on inequality constraints */
#define SET_ACC_RESINEQ_MPCC_Car	(MPCC_Car_float)(1E-06)

/* desired maximum violation of complementarity */
#define SET_ACC_KKTCOMPL_MPCC_Car	(MPCC_Car_float)(1E-06)


/* SOLVER RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define OPTIMAL_MPCC_Car      (1)

/* maximum number of iterations has been reached */
#define MAXITREACHED_MPCC_Car (0)

/* solver has stopped due to a timeout */
#define TIMEOUT_MPCC_Car   (2)

/* wrong number of inequalities error */
#define INVALID_NUM_INEQ_ERROR_MPCC_Car  (-4)

/* factorization error */
#define FACTORIZATION_ERROR_MPCC_Car   (-5)

/* NaN encountered in function evaluations */
#define BADFUNCEVAL_MPCC_Car  (-6)

/* no progress in method possible */
#define NOPROGRESS_MPCC_Car   (-7)

/* invalid values in parameters */
#define PARAM_VALUE_ERROR_MPCC_Car   (-11)

/* too small timeout given */
#define INVALID_TIMEOUT_MPCC_Car   (-12)

/* licensing error - solver not valid on this machine */
#define LICENSE_ERROR_MPCC_Car  (-100)

/* INTEGRATORS RETURN CODE ------------*/
/* Integrator ran successfully */
#define INTEGRATOR_SUCCESS (11)
/* Number of steps set by user exceeds maximum number of steps allowed */
#define INTEGRATOR_MAXSTEPS_EXCEEDED (12)



/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct
{
    /* vector of size 18 */
    MPCC_Car_float xinit[18];

    /* vector of size 806 */
    MPCC_Car_float x0[806];

    /* vector of size 3348 */
    MPCC_Car_float all_parameters[3348];


} MPCC_Car_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct
{
    /* vector of size 806 */
    MPCC_Car_float all_var[806];


} MPCC_Car_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct
{
    /* iteration number */
    solver_int32_default it;

	/* number of iterations needed to optimality (branch-and-bound) */
	solver_int32_default it2opt;
	
    /* inf-norm of equality constraint residuals */
    MPCC_Car_float res_eq;
	
    /* inf-norm of inequality constraint residuals */
    MPCC_Car_float res_ineq;

	/* norm of stationarity condition */
    MPCC_Car_float rsnorm;

	/* max of all complementarity violations */
    MPCC_Car_float rcompnorm;

    /* primal objective */
    MPCC_Car_float pobj;	
	
    /* dual objective */
    MPCC_Car_float dobj;	

    /* duality gap := pobj - dobj */
    MPCC_Car_float dgap;		
	
    /* relative duality gap := |dgap / pobj | */
    MPCC_Car_float rdgap;		

    /* duality measure */
    MPCC_Car_float mu;

	/* duality measure (after affine step) */
    MPCC_Car_float mu_aff;
	
    /* centering parameter */
    MPCC_Car_float sigma;
	
    /* number of backtracking line search steps (affine direction) */
    solver_int32_default lsit_aff;
    
    /* number of backtracking line search steps (combined direction) */
    solver_int32_default lsit_cc;
    
    /* step size (affine direction) */
    MPCC_Car_float step_aff;
    
    /* step size (combined direction) */
    MPCC_Car_float step_cc;    

	/* solvertime */
	MPCC_Car_float solvetime;   

	/* time spent in function evaluations */
	MPCC_Car_float fevalstime;  


} MPCC_Car_info;







/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* Time of Solver Generation: (UTC) Thursday, February 11, 2021 1:07:50 PM */
/* User License expires on: (UTC) Saturday, August 14, 2021 10:00:00 PM (approx.) (at the time of code generation) */
/* Solver Static License expires on: (UTC) Saturday, August 14, 2021 10:00:00 PM (approx.) */
/* Solver Generation Request Id: 780875bf-8eec-4494-899a-3d6421d6dc29 */
/* examine exitflag before using the result! */
#ifdef __cplusplus
extern "C" {
#endif		

typedef void (*MPCC_Car_extfunc)(MPCC_Car_float* x, MPCC_Car_float* y, MPCC_Car_float* lambda, MPCC_Car_float* params, MPCC_Car_float* pobj, MPCC_Car_float* g, MPCC_Car_float* c, MPCC_Car_float* Jeq, MPCC_Car_float* h, MPCC_Car_float* Jineq, MPCC_Car_float* H, solver_int32_default stage, solver_int32_default iterations, solver_int32_default threadID);

extern solver_int32_default MPCC_Car_solve(MPCC_Car_params *params, MPCC_Car_output *output, MPCC_Car_info *info, FILE *fs, MPCC_Car_extfunc evalextfunctions_MPCC_Car);	







#ifdef __cplusplus
}
#endif

#endif
