/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "laplace2d.hpp"
#include <memory>
#include <cublas_v2.h>

#define OFFSET(x, y, m) (((x) * (m)) + (y))

Laplace::Laplace(int m, int n, InitFunc initFunc) : m(m), n(n)
{
    std::unique_ptr<double[]> A_ptr(new double[n * m]);
    std::unique_ptr<double[]> Anew_ptr(new double[n * m]);

    std::memset(A_ptr.get(), 0, n * m * sizeof(double));
    std::memset(Anew_ptr.get(), 0, n * m * sizeof(double));

    initFunc(A_ptr, Anew_ptr, n, m);

    A = A_ptr.get(), Anew = Anew_ptr.get();

#pragma acc enter data copyin(this)
#pragma acc enter data copyin(A[ : m * n], Anew[ : m * n])
}

Laplace::~Laplace()
{
#pragma acc exit data delete (A[ : m * n], Anew[ : m * n])
#pragma acc exit data delete (this)
}

void Laplace::save()
{
    std::ofstream out("out.txt");

    out << std::fixed << std::setprecision(5);

#pragma acc update host(A[ : n * m])
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            out << std::left << std::setw(10) << A[OFFSET(j, i, m)] << " ";
        }
        out << std::endl;
    }
}

void Laplace::calcNext()
{
#pragma acc parallel loop present(A, Anew)
    for (int j = 1; j < n - 1; j++)
    {
#pragma acc loop
        for (int i = 1; i < m - 1; i++)
        {
            Anew[OFFSET(j, i, m)] = 0.25 * (A[OFFSET(j, i + 1, m)] + A[OFFSET(j, i - 1, m)] + A[OFFSET(j - 1, i, m)] + A[OFFSET(j + 1, i, m)]);
        }
    }
}

double Laplace::calcError()
{
    cublasHandle_t handler;
    cublasStatus_t status;

    status = cublasCreate(&handler);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cublasCreate failed with error code: " << status << std::endl;
        return 13;
    }
    int idx = 0;
    double alpha = -1.0, error = 1.0;



#pragma acc host_data use_device(A, Anew)
    status = cublasDaxpy(handler, n * m, &alpha, Anew, 1, A, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cublasDaxpy failed with error code: " << status << std::endl;
        exit(1);
    }

    status = cublasIdamax(handler, n * n, A, 1, &idx);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cublasIdamax failed with error code: " << status << std::endl;
        exit(1);
    }

    #pragma acc update host(A[idx - 1]) 
    error = std::fabs(A[idx - 1]);

    #pragma acc host_data use_device(A, Anew)
    status = cublasDcopy(handler, n * n, Anew, 1, A, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasDcopy failed with error code: " << status << std::endl;
        exit (1);
    }

    cublasDestroy(handler);

    return error;
}

void Laplace::swap()
{
    double *temp = A;
    A = Anew;
    Anew = temp;
#pragma acc data present(A, Anew)

    return;
}
