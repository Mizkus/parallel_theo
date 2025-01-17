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
#include <omp.h>

#define OFFSET(x, y, m) (((x) * (m)) + (y))

Laplace::Laplace(int m, int n, InitFunc initFunc) : m(m), n(n)
{
    A = new double[n * m];
    Anew = new double[n * m];

    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));

    initFunc(A, Anew, n, m);

#pragma acc enter data copyin(this)
#pragma acc enter data copyin(A[ : m * n], Anew[ : m * n])
}

Laplace::~Laplace()
{
#pragma acc exit data delete (A[:m * n], Anew[:m * n])
#pragma acc exit data delete (this)

    delete[] A;
    delete[] Anew;
}

void Laplace::save()
{
    std::ofstream out("out.txt");

    out << std::fixed << std::setprecision(5);

#pragma acc update host(A[:n * m])
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
    for (int j = 1; j < n  - 1; j++)
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
    double error = 0.0;
#pragma acc parallel loop reduction(max : error) present(A, Anew)
    for (int j = 1; j < n - 1; j++)
    {
#pragma acc loop 
        for (int i = 1; i < m - 1; i++)
        {
            error = fmax(error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i, m)]));
        }
    }
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
