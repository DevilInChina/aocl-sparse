/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef TESTING_CSRMV_HPP
#define TESTING_CSRMV_HPP

#include "aoclsparse.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_check.hpp"
#include "utility.hpp"
#include "aoclsparse_random.hpp"
#include <cstdio>
template <typename T>
void testing_csrmv(const Arguments& arg)
{
    aoclsparse_int         M         = arg.M;
    aoclsparse_int         N         = arg.N;
    aoclsparse_int         nnz       = arg.nnz;
    aoclsparse_operation   trans     = arg.transA;
    aoclsparse_index_base  base      = arg.baseA;
    aoclsparse_matrix_init mat       = arg.matrix;
    std::string           filename = arg.filename; 

    T h_alpha = static_cast<T>(arg.alpha);
    T h_beta  = static_cast<T>(arg.beta);

    // Create matrix descriptor
    aoclsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, base));

    // Allocate memory for matrix
    std::vector<aoclsparse_int> hcsr_row_ptr;
    std::vector<aoclsparse_int> hcsr_col_ind;
    std::vector<T>             hcsr_val;

    aoclsparse_seedrand();
#if 0
    // Print aoclsparse version
    int  ver;

    aoclsparse_get_version(&ver);

    std::cout << "aocl-sparse version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << std::endl;
#endif
    // Sample matrix
    aoclsparse_init_csr_matrix(hcsr_row_ptr,
                              hcsr_col_ind,
                              hcsr_val,
                              M,
                              N,
                              nnz,
                              base,
                              mat,
                              filename.c_str(),
                              false,
                              false);

    // Allocate memory for vectors
    std::vector<T> hx(N);
    std::vector<T> hy(M);
    std::vector<T> hy_gold(M);

    // Initialize data
    aoclsparse_init<T>(hx, 1, N, 1);
    aoclsparse_init<T>(hy, 1, M, 1);
    hy_gold = hy; 
    if(arg.unit_check)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_csrmv(trans,
                                                 &h_alpha,
                                                 M,
                                                 N,
                                                 nnz,
                                                 hcsr_val.data(),
                                                 hcsr_col_ind.data(),
                                                 hcsr_row_ptr.data(),
                                                 descr,
                                                 hx.data(),
                                                 &h_beta,
                                                 hy.data()));
	// Reference SPMV CSR implementation
        for(int i = 0; i < M; i++)
        {
            T result = 0.0;
            for(int j = hcsr_row_ptr[i]-base ; j < hcsr_row_ptr[i+1]-base ; j++)
     	    {
                result += h_alpha * hcsr_val[j] * hx[hcsr_col_ind[j] - base];
	        }
            hy_gold[i] = (h_beta * hy_gold[i]) + result;
        }
        near_check_general<T>(1, M, 1, hy_gold.data(), hy.data());
    }
    int number_hot_calls  = arg.iters;

    for(int threads = 64 ; threads <= 256 ; threads<<=1) {
        omp_set_num_threads(threads);
        double cpu_time_used = 0;
        double min_time = 1e9;
        // Performance run
        for (int iter = 0; iter < number_hot_calls; ++iter) {
            double cpu_time_start = get_time_us();
            CHECK_AOCLSPARSE_ERROR(aoclsparse_csrmv(trans,
                                                    &h_alpha,
                                                    M,
                                                    N,
                                                    nnz,
                                                    hcsr_val.data(),
                                                    hcsr_col_ind.data(),
                                                    hcsr_row_ptr.data(),
                                                    descr,
                                                    hx.data(),
                                                    &h_beta,
                                                    hy.data()));
            double cpu_time_stop = get_time_us();
            double time_current = (cpu_time_stop - cpu_time_start);
            cpu_time_used += time_current;
            min_time = std::min(min_time, time_current);
        }


        cpu_time_used /= number_hot_calls;
        double cpu_gflops
                = spmv_gflop_count<T>(M, nnz, h_beta != static_cast<T>(0)) / cpu_time_used * 1e6;

        double cpu_gflops_max
                = spmv_gflop_count<T>(M, nnz, h_beta != static_cast<T>(0)) / min_time * 1e6;
        printf("%s,AOCL_SPMV,AVX_ON_CHOOSE,%d,%d,%f,%f,%f,%f,%f\n", arg.filename
                , threads, nnz, 0.0,
               0.0, cpu_time_used, cpu_gflops,cpu_gflops_max);
    }
} 

#endif // TESTING_CSRMV_HPP
