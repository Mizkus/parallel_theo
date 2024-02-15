#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cmath>

#define N_STEPS 40000

#define A -4.0
#define B 4.0

double func(double x)
{
    return exp(-x * x);
}

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double integrate_omp(double (*func)(double), double a, double b, int n, size_t num_threads)
{
    double h = (b - a) / n;
    double sum = 0.0;
#pragma omp parallel num_threads(num_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sumloc = 0.0;
        for (int i = lb; i <= ub; i++)
            sumloc += func(a + h * (i + 0.5));
#pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}

int main(int argc, char *argv[])
{
    size_t num_threads = 0;
    if (argc > 1)
         num_threads = atoi(argv[1]);


    double t = cpuSecond();
    integrate_omp(func, A, B, N_STEPS, num_threads);
    t = cpuSecond() - t;

    printf("Elapsed time (parallel): %.6f sec.\n", t);

    return 0;
}