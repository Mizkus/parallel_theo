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
#include "laplace2d.hpp"
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>

#include <chrono>
#include <iostream>

void initFunc(double* A, double* Anew, int n, int m){


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
    desc.add_options()
        ("help", "help desciption")
        ("n", po::value<int>(&n), "int")
        ("iter", po::value<int>(&iter_max), "int")
        ("err", po::value<double>(&tol), "double");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
    }

    m = n;


    double error = 1.0;

    Laplace a(n, m, initFunc);

    nvtxRangePushA("init");
    nvtxRangePop();
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);

    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;

    nvtxRangePushA("while");
    while (error > tol && iter < iter_max)
    {
        nvtxRangePushA("calc");
        a.calcNext();
        nvtxRangePop();

        if (iter % 100 == 0){
            error = a.calcError();
            printf("%5d, %0.6f\n", iter, error);
        }


        nvtxRangePushA("swap");
        a.swap();
        nvtxRangePop();
        
        iter++;
    }
    nvtxRangePop();

    a.save();

    auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << "TIME: " << runtime.count() / 1000000.;
    
    return 0;
}