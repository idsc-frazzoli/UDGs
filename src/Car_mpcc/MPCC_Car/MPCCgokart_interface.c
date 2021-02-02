/*
 * CasADi to FORCES Template - missing information to be filled in by createCasadi.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2020. All rights reserved.
 *
 * This file is part of the FORCES client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif
    
#include "include/MPCCgokart.h"

#define casadi_real MPCCgokart_float

#include "MPCCgokart_model.h"
 

   

/* copies data from sparse matrix into a dense one */
static void sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, MPCCgokart_float *data, MPCCgokart_float *out)
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
extern void MPCCgokart_casadi2forces(MPCCgokart_float *x,        /* primal vars                                         */
                                 MPCCgokart_float *y,        /* eq. constraint multiplers                           */
                                 MPCCgokart_float *l,        /* ineq. constraint multipliers                        */
                                 MPCCgokart_float *p,        /* parameters                                          */
                                 MPCCgokart_float *f,        /* objective function (scalar)                         */
                                 MPCCgokart_float *nabla_f,  /* gradient of objective function                      */
                                 MPCCgokart_float *c,        /* dynamics                                            */
                                 MPCCgokart_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 MPCCgokart_float *h,        /* inequality constraints                              */
                                 MPCCgokart_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 MPCCgokart_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                            */
								 solver_int32_default iteration, /* iteration number of solver                          */
								 solver_int32_default threadID  /* Id of caller thread 								   */)
{
    /* CasADi input and output arrays */
    const MPCCgokart_float *in[4];
    MPCCgokart_float *out[7];
	

	/* Allocate working arrays for CasADi */
	MPCCgokart_float w[510];
	
    /* temporary storage for casadi sparse output */
    MPCCgokart_float this_f;
    MPCCgokart_float nabla_f_sparse[7];
    MPCCgokart_float h_sparse[3];
    MPCCgokart_float nabla_h_sparse[10];
    MPCCgokart_float c_sparse[8];
    MPCCgokart_float nabla_c_sparse[30];
            
    
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
		MPCCgokart_objective_0(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = MPCCgokart_objective_0_sparsity_out(1)[0];
			ncol = MPCCgokart_objective_0_sparsity_out(1)[1];
			colind = MPCCgokart_objective_0_sparsity_out(1) + 2;
			row = MPCCgokart_objective_0_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = c_sparse;
		out[1] = nabla_c_sparse;
		MPCCgokart_dynamics_0(in, out, NULL, w, 0);
		if( c )
		{
			nrow = MPCCgokart_dynamics_0_sparsity_out(0)[0];
			ncol = MPCCgokart_dynamics_0_sparsity_out(0)[1];
			colind = MPCCgokart_dynamics_0_sparsity_out(0) + 2;
			row = MPCCgokart_dynamics_0_sparsity_out(0) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, c_sparse, c);
		}
		if( nabla_c )
		{
			nrow = MPCCgokart_dynamics_0_sparsity_out(1)[0];
			ncol = MPCCgokart_dynamics_0_sparsity_out(1)[1];
			colind = MPCCgokart_dynamics_0_sparsity_out(1) + 2;
			row = MPCCgokart_dynamics_0_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_c_sparse, nabla_c);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		MPCCgokart_inequalities_0(in, out, NULL, w, 0);
		if( h )
		{
			nrow = MPCCgokart_inequalities_0_sparsity_out(0)[0];
			ncol = MPCCgokart_inequalities_0_sparsity_out(0)[1];
			colind = MPCCgokart_inequalities_0_sparsity_out(0) + 2;
			row = MPCCgokart_inequalities_0_sparsity_out(0) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = MPCCgokart_inequalities_0_sparsity_out(1)[0];
			ncol = MPCCgokart_inequalities_0_sparsity_out(1)[1];
			colind = MPCCgokart_inequalities_0_sparsity_out(1) + 2;
			row = MPCCgokart_inequalities_0_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
	if ((30 == stage))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		MPCCgokart_objective_1(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = MPCCgokart_objective_1_sparsity_out(1)[0];
			ncol = MPCCgokart_objective_1_sparsity_out(1)[1];
			colind = MPCCgokart_objective_1_sparsity_out(1) + 2;
			row = MPCCgokart_objective_1_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		MPCCgokart_inequalities_1(in, out, NULL, w, 0);
		if( h )
		{
			nrow = MPCCgokart_inequalities_1_sparsity_out(0)[0];
			ncol = MPCCgokart_inequalities_1_sparsity_out(0)[1];
			colind = MPCCgokart_inequalities_1_sparsity_out(0) + 2;
			row = MPCCgokart_inequalities_1_sparsity_out(0) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = MPCCgokart_inequalities_1_sparsity_out(1)[0];
			ncol = MPCCgokart_inequalities_1_sparsity_out(1)[1];
			colind = MPCCgokart_inequalities_1_sparsity_out(1) + 2;
			row = MPCCgokart_inequalities_1_sparsity_out(1) + 2 + (ncol + 1);
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