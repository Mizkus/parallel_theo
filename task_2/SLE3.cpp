#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifndef NUM_THREADS
#define NUM_THREADS 40
#endif

#ifndef CHUNCK_SIZE
#define CHUNCK_SIZE 20
#endif

#ifndef TYPE
#define TYPE static
#endif

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void init_matrix(double *matrix, int n)
{
#pragma omp parallel for  num_threads(NUM_THREADS) schedule(TYPE, CHUNCK_SIZE)    
for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                matrix[i * n + j] = 2.0;
            else
                matrix[i * n + j] = 1.0;
        }
    }
}

void matrix_vector_product(double *mx, double *vec, double *res, int n)
{
#pragma omp parallel for  num_threads(NUM_THREADS) schedule(TYPE, CHUNCK_SIZE) 
    for (int i = 0; i < n; i++)
    {
        int sum = 0;
        for (int j = 0; j < n; j++)
        {
            sum += mx[i * n + j] * vec[j];
        }
        res[i] = sum;
    }
}

void vector_sub(double *vec_1, double *vec_2, double *res, int n)
{
#pragma omp parallel for  num_threads(NUM_THREADS) schedule(TYPE, CHUNCK_SIZE)     
for (int i = 0; i < n; i++)
    {
        res[i] = vec_1[i] - vec_2[i];
    }
}

void inplace_vector_sub(double *vec_1, double *vec_2, int n)
{
#pragma omp parallel for  num_threads(NUM_THREADS) schedule(TYPE, CHUNCK_SIZE)     
for (int i = 0; i < n; i++)
    {
        vec_1[i] -= vec_2[i];
    }
}

void vector_scalar_product(double *vec, double scale, int n)
{
#pragma omp parallel for  num_threads(NUM_THREADS) schedule(TYPE, CHUNCK_SIZE)     
for (int i = 0; i < n; i++)
    {
        vec[i] *= scale;
    }
}

double find_norm(double *vec, int n)
{
    double norm = 0;
    double sumloc = 0.0;
#pragma omp parallel for  num_threads(NUM_THREADS) schedule(TYPE, CHUNCK_SIZE)     
for (int i = 0; i < n; i++)
    {
        norm += pow(vec[i], 2);
    }
#pragma omp atomic
    norm += sumloc;
    return sqrt(norm);
}

void simple_iteration(double *matrix, double *x, double *b, int n, double tau, double eps)
{
    double *ax = (double *)malloc(n * sizeof(double));
    double *subs = (double *)malloc(n * sizeof(double));
    double norm_b = find_norm(b, n);
    double norm_sub;

        while (1)
        {

            matrix_vector_product(matrix, x, ax, n);
            vector_sub(ax, b, subs, n);
            norm_sub = find_norm(subs, n);
            if (norm_sub / norm_b < eps)
                break;
            vector_scalar_product(subs, tau, n);
            inplace_vector_sub(x, subs, n);
        }

    free(ax);
    free(subs);

}

int main(int argc, char **argv)
{
    int n = 1000;

    
    if (argc > 1)
        n = atoi(argv[1]);

    double tau = 0.0000025;
    if (argc > 2)
        tau = atof(argv[2]);

    double eps = 0.00001;
    if (argc > 3)
        eps = atof(argv[3]);

    double *matrix = (double *)calloc(n * n, sizeof(double *));

    double *x = (double *)calloc(n, sizeof(double));

    double *b = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        b[i] = n + 1;
    }
    init_matrix(matrix, n);

    double t = cpuSecond();
    simple_iteration(matrix, x, b, n, tau, eps);
    t = cpuSecond() - t;

    free(matrix);
    free(x);
    free(b);

    printf("%.6f\n", t);

    return 0;
}