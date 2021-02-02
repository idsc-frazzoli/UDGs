/*
 * CasADi to FORCES Template - missing information to be filled in by createCasadi.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2020. All rights reserved.
 *
 * This file is part of the FORCES client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif
    
#include "include/MPCCcar.h"

#define casadi_real MPCCcar_float

#include "MPCCcar_model.h"
 

   

/* copies data from sparse matrix into a dense one */
static void sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, MPCCcar_float *data, MPCCcar_float *out)
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
extern void MPCCcar_casadi2forces(MPCCcar_float *x,        /* primal vars                                         */
                                 MPCCcar_float *y,        /* eq. constraint multiplers                           */
                                 MPCCcar_float *l,        /* ineq. constraint multipliers                        */
                                 MPCCcar_float *p,        /* parameters                                          */
                                 MPCCcar_float *f,        /* objective function (scalar)                         */
                                 MPCCcar_float *nabla_f,  /* gradient of objective function                      */
                                 MPCCcar_float *c,        /* dynamics                                            */
                                 MPCCcar_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 MPCCcar_float *h,        /* inequality constraints                              */
                                 MPCCcar_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 MPCCcar_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                            */
								 solver_int32_default iteration, /* iteration number of solver                          */
								 solver_int32_default threadID  /* Id of caller thread 								   */)
{
    /* CasADi input and output arrays */
    const MPCCcar_float *in[4];
    MPCCcar_float *out[7];
	

	/* Allocate working arrays for CasADi */
	MPCCcar_float w[1044];
	
    /* temporary storage for casadi sparse output */
    MPCCcar_float this_f;
    MPCCcar_float nabla_f_sparse[7];
    MPCCcar_float h_sparse[3];
    MPCCcar_float nabla_h_sparse[8];
    MPCCcar_float c_sparse[9];
    MPCCcar_float nabla_c_sparse[42];
            
    
    /* pointers to row and column info for 
     * column compressed format used by CasADi */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    /* set inputs for CasADi */
    in[0] = x;
    in[1] = p; /* maybe should be made conditional */
    in[2] = l; /* maybe should be made conditional */     
    in[3] = y; /* maybe should be made conditional */


	if ((0 <= stage && stage <= 58))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		MPCCcar_objective_0(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = MPCCcar_objective_0_sparsity_out(1)[0];
			ncol = MPCCcar_objective_0_sparsity_out(1)[1];
			colind = MPCCcar_objective_0_sparsity_out(1) + 2;
			row = MPCCcar_objective_0_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = c_sparse;
		out[1] = nabla_c_sparse;
		MPCCcar_dynamics_0(in, out, NULL, w, 0);
		if( c )
		{
			nrow = MPCCcar_dynamics_0_sparsity_out(0)[0];
			ncol = MPCCcar_dynamics_0_sparsity_out(0)[1];
			colind = MPCCcar_dynamics_0_sparsity_out(0) + 2;
			row = MPCCcar_dynamics_0_sparsity_out(0) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, c_sparse, c);
		}
		if( nabla_c )
		{
			nrow = MPCCcar_dynamics_0_sparsity_out(1)[0];
			ncol = MPCCcar_dynamics_0_sparsity_out(1)[1];
			colind = MPCCcar_dynamics_0_sparsity_out(1) + 2;
			row = MPCCcar_dynamics_0_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_c_sparse, nabla_c);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		MPCCcar_inequalities_0(in, out, NULL, w, 0);
		if( h )
		{
			nrow = MPCCcar_inequalities_0_sparsity_out(0)[0];
			ncol = MPCCcar_inequalities_0_sparsity_out(0)[1];
			colind = MPCCcar_inequalities_0_sparsity_out(0) + 2;
			row = MPCCcar_inequalities_0_sparsity_out(0) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = MPCCcar_inequalities_0_sparsity_out(1)[0];
			ncol = MPCCcar_inequalities_0_sparsity_out(1)[1];
			colind = MPCCcar_inequalities_0_sparsity_out(1) + 2;
			row = MPCCcar_inequalities_0_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
	if ((59 == stage))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		MPCCcar_objective_1(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = MPCCcar_objective_1_sparsity_out(1)[0];
			ncol = MPCCcar_objective_1_sparsity_out(1)[1];
			colind = MPCCcar_objective_1_sparsity_out(1) + 2;
			row = MPCCcar_objective_1_sparsity_out(1) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		MPCCcar_inequalities_1(in, out, NULL, w, 0);
		if( h )
		{
			nrow = MPCCcar_inequalities_1_sparsity_out(0)[0];
			ncol = MPCCcar_inequalities_1_sparsity_out(0)[1];
			colind = MPCCcar_inequalities_1_sparsity_out(0) + 2;
			row = MPCCcar_inequalities_1_sparsity_out(0) + 2 + (ncol + 1);
			sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = MPCCcar_inequalities_1_sparsity_out(1)[0];
			ncol = MPCCcar_inequalities_1_sparsity_out(1)[1];
			colind = MPCCcar_inequalities_1_sparsity_out(1) + 2;
			row = MPCCcar_inequalities_1_sparsity_out(1) + 2 + (ncol + 1);
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