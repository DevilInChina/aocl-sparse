/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 * ************************************************************************ */

#include "aoclsparse.h"
#include "aoclsparse_csrmv.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
extern "C" aoclsparse_status aoclsparse_dcsrmv(aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
				   const double*              alpha,
                                   const double*              csr_val,
                                   const aoclsparse_int*      csr_row_ptr,
                                   const aoclsparse_int*      csr_col_ind,
                                   const double*             x,
                                   const double*             beta,
                                   double*                   y
)
{
    return aoclsparse_csrmv(m,
                            n,
                            nnz,
                            alpha,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            x,
                            beta,
                            y);
}

extern "C" aoclsparse_status aoclsparse_scsrmv(aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
				   const float*              alpha,
                                   const float*              csr_val,
                                   const aoclsparse_int*      csr_row_ptr,
                                   const aoclsparse_int*      csr_col_ind,
                                   const float*             x,
                                   const float*             beta,
                                   float*                   y
)
{
    return aoclsparse_csrmv(m,
                            n,
                            nnz,
                            alpha,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            x,
                            beta,
                            y);
}
