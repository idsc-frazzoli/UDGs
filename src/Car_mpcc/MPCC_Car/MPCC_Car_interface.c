/*
 * CasADi to FORCES Template - missing information to be filled in by createCasadi.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2020. All rights reserved.
 *
 * This file is part of the FORCES client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif
    
#include "include/MPCC_Car.h"

#define casadi_real MPCC_Car_float

#include "MPCC_Car_model.h"
 

   

/* copies data from sparse matrix into a dense one */
static void sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, MPCC_Car_float *data, MPCC_Car_float *out)
{
    solver_int32_default i, j;
    
    /* copy data into dense matrix */
    for(i=0; i<ncol; i++)
    {
        for( j=colidx[i]; j < colidx[i+1]; j++ )
        {
            out[i*nrow + row[j]] = data[j];
        }
    }
}




/* CasADi - FORCES interface */
extern void MPCC_Car_casadi2forces(MPCC_Car_float *x,        /* primal vars                                         */
                                 MPCC_Car_float *y,        /* eq. constraint multiplers                           */
                                 MPCC_Car_float *l,        /* ineq. constraint multipliers                        */
                                 MPCC_Car_float *p,        /* parameters                                          */
                                 MPCC_Car_float *f,        /* objective function (scalar)                         */
                                 MPCC_Car_float *nabla_f,  /* gradient of objective function                      */
                                 MPCC_Car_float *c,        /* dynamics                                            */
                                 MPCC_Car_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 MPCC_Car_float *h,        /* inequality constraints                              */
                                 MPCC_Car_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 MPCC_Car_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                            */
								 solver_int32_default iteration, /* iteration number of solver                          */
								 solver_int32_default threadID  /* Id of caller thread 								   */)
{
    /* CasADi input and output arrays */
    const MPCC_Car_float *in[4];
    MPCC_Car_float *out[7];
	

	/* Allocate working arrays for CasADi */
	MPCC_Car_float w[2053];
	
    /* temporary storage for casadi sparse output */
    MPCC_Car_float this_f;
    MPCC_Car_float nabla_f_sparse[14];
    MPCC_Car_float h_sparse[6];
    MPCC_Car_float nabla_h_sparse[18];
    MPCC_Car_float c_sparse[18];
    MPCC_Car_float nabla_c_sparse[84];
            
    
    /* pointers to row and column info for 
     * column compressed format used by CasADi */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    /* set inputs for CasADi */
    in[0] = x;
    in[1] = p; /* maybe should be made conditional */
    in[2] = l; /* maybe should be made conditional */     
    in[3] = y; /* maybe should be made conditional */


	if ((0 <= stage && stage <= 29))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		MPCC_Car_objective_0(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = MPCC_Car_objective_0_sparsity_out(1)[0];
			ncol = MPCC_Car_objective_0_sparsity_out(1)[1];
			colind = MPCC_Car_objective_0_sparsity_out(1) + 2;
			row = MPCC_Car_objective_0_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = c_sparse;
		out[1] = nabla_c_sparse;
		MPCC_Car_dynamics_0(in, out, NULL, w, 0);
		if( c )
		{
			nrow = MPCC_Car_dynamics_0_sparsity_out(0)[0];
			ncol = MPCC_Car_dynamics_0_sparsity_out(0)[1];
			colind = MPCC_Car_dynamics_0_sparsity_out(0) + 2;
			row = MPCC_Car_dynamics_0_sparsity_out(0) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, c_sparse, c);
		}
		if( nabla_c )
		{
			nrow = MPCC_Car_dynamics_0_sparsity_out(1)[0];
			ncol = MPCC_Car_dynamics_0_sparsity_out(1)[1];
			colind = MPCC_Car_dynamics_0_sparsity_out(1) + 2;
			row = MPCC_Car_dynamics_0_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_c_sparse, nabla_c);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		MPCC_Car_inequalities_0(in, out, NULL, w, 0);
		if( h )
		{
			nrow = MPCC_Car_inequalities_0_sparsity_out(0)[0];
			ncol = MPCC_Car_inequalities_0_sparsity_out(0)[1];
			colind = MPCC_Car_inequalities_0_sparsity_out(0) + 2;
			row = MPCC_Car_inequalities_0_sparsity_out(0) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = MPCC_Car_inequalities_0_sparsity_out(1)[0];
			ncol = MPCC_Car_inequalities_0_sparsity_out(1)[1];
			colind = MPCC_Car_inequalities_0_sparsity_out(1) + 2;
			row = MPCC_Car_inequalities_0_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
	if ((30 == stage))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		MPCC_Car_objective_1(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = MPCC_Car_objective_1_sparsity_out(1)[0];
			ncol = MPCC_Car_objective_1_sparsity_out(1)[1];
			colind = MPCC_Car_objective_1_sparsity_out(1) + 2;
			row = MPCC_Car_objective_1_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		MPCC_Car_inequalities_1(in, out, NULL, w, 0);
		if( h )
		{
			nrow = MPCC_Car_inequalities_1_sparsity_out(0)[0];
			ncol = MPCC_Car_inequalities_1_sparsity_out(0)[1];
			colind = MPCC_Car_inequalities_1_sparsity_out(0) + 2;
			row = MPCC_Car_inequalities_1_sparsity_out(0) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = MPCC_Car_inequalities_1_sparsity_out(1)[0];
			ncol = MPCC_Car_inequalities_1_sparsity_out(1)[1];
			colind = MPCC_Car_inequalities_1_sparsity_out(1) + 2;
			row = MPCC_Car_inequalities_1_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}         
    
    /* add to objective */
    if( f )
    {
        *f += this_f;
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif