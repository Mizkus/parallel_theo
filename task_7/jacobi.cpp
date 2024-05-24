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

#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>

#include <chrono>
#include <iostream>

#include <functional>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cublas_v2.h>

#define OFFSET(x, y, m) (((x) * (m)) + (y))

void initFunc(std::unique_ptr<double[]>& A,  std::unique_ptr<double[]>& Anew, int n, int m){


    double corners[4] = {10, 20, 30, 20};
    double step = (corners[1] - corners[0]) / (n - 1);

    int lastIdx = n - 1;
    A[0] = Anew[0] = corners[0];
    A[lastIdx] = Anew[lastIdx] = corners[1];
    A[n * lastIdx] = Anew[n * lastIdx] = corners[3];
    A[n * n - 1] = Anew[n * n - 1] = corners[2];

    for (int i = 1; i < n - 1; i++)
    {
        double val = corners[0] + i * step;
        A[i] = Anew[i] = val;                       
        A[n * i] = Anew[n * i] = val;               
        A[lastIdx + n * i] = Anew[lastIdx + n * i] = corners[1] + i * step;
        A[n * lastIdx + i] = Anew[n * lastIdx + i] = corners[3] + i * step; 
    }
}

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    int n = 128, m = 128;
    double tol = 1.0e-6;
    int iter_max = 100;

    po::options_description desc("Allowed options");
    desc.add_options()("help", "help desciption")("n", po::value<int>(&n), "int")("iter", po::value<int>(&iter_max), "int")("err", po::value<double>(&tol), "double");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    po::notify(vm);

    m = n;

    double error = 1.0;

    std::unique_ptr<double[]> A_ptr(new double[n * m]);
    std::unique_ptr<double[]> Anew_ptr(new double[n * m]);

    std::memset(A_ptr.get(), 0, n * m * sizeof(double));
    std::memset(Anew_ptr.get(), 0, n * m * sizeof(double));

    initFunc(A_ptr, Anew_ptr, n, m);

    double* A = A_ptr.get(), *Anew = Anew_ptr.get();

#pragma acc enter data copyin(A[ : m * n], Anew[ : m * n])

    cublasHandle_t handler;
    cublasStatus_t status;

    status = cublasCreate(&handler);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cublasCreate failed with error code: " << status << std::endl;
        exit(1);
    }

    nvtxRangePushA("init");
    nvtxRangePop();
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);

    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;

    nvtxRangePushA("while");
    while (error > tol && iter < iter_max)
    {
        nvtxRangePushA("calc");
#pragma acc parallel loop collapse(2) present(A, Anew)
        for (int j = 1; j < n - 1; j++)
        {
            for (int i = 1; i < m - 1; i++)
            {
                Anew[OFFSET(j, i, m)] = (A[OFFSET(j, i + 1, m)] + A[OFFSET(j, i - 1, m)] + A[OFFSET(j - 1, i, m)] + A[OFFSET(j + 1, i, m)]) * 0.25;
            }
        }
        nvtxRangePop();

        if (iter % 1000 == 0)
        {
            int idx = 0;
            double alpha = -1.0;
            error = 1.0;

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
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                std::cerr << "cublasDcopy failed with error code: " << status << std::endl;
                exit(1);
            }
            printf("%5d, %0.6f\n", iter, error);
        }

        nvtxRangePushA("swap");
        double *temp = A;
        A = Anew;
        Anew = temp;
        nvtxRangePop();

        iter++;
    }
    nvtxRangePop();

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

    auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << "TIME: " << runtime.count() / 1000000.;

#pragma acc exit data delete (A[ : m * n], Anew[ : m * n])

    cublasDestroy(handler);


    return 0;
}