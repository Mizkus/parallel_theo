#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace po = boost::program_options;

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, std::function<void(T *)>>;

double *cuda_new(size_t size)
{
    double *d_ptr;
    cudaError_t cudaErr = cudaMalloc((void **)&d_ptr, sizeof(double) * size);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "Memory allocation error: " << cudaGetErrorString(cudaErr) << std::endl;
        exit(1);
    }
    return d_ptr;
}

void cuda_delete(double *dev_ptr)
{
    cudaFree(dev_ptr);
}

cudaStream_t *cuda_new_stream()
{
    cudaStream_t *stream = new cudaStream_t;
    cudaStreamCreate(stream);
    return stream;
}

void cuda_delete_stream(cudaStream_t *stream)
{
    cudaStreamDestroy(*stream);
    delete stream;
}

cudaGraph_t *cuda_new_graph()
{
    cudaGraph_t *graph = new cudaGraph_t;
    return graph;
}

void cuda_delete_graph(cudaGraph_t *graph)
{
    cudaGraphDestroy(*graph);
    delete graph;
}

cudaGraphExec_t *cuda_new_graph_save()
{
    cudaGraphExec_t *graphExec = new cudaGraphExec_t;
    return graphExec;
}

void cuda_delete_graph_save(cudaGraphExec_t *graphExec)
{
    cudaGraphExecDestroy(*graphExec);
    delete graphExec;
}

void initialize(std::unique_ptr<double[]> &A, std::unique_ptr<double[]> &Anew, int n)
{
    memset(A.get(), 0, n * n * sizeof(double));

    double corners[4] = {10, 20, 30, 20};
    A[0] = corners[0];
    A[n - 1] = corners[1];
    A[n * n - 1] = corners[2];
    A[n * (n - 1)] = corners[3];
    double step = (corners[1] - corners[0]) / (n - 1);

    for (int i = 1; i < n - 1; i++)
    {
        A[i] = corners[0] + i * step;
        A[n * i] = corners[0] + i * step;
        A[(n - 1) + n * i] = corners[1] + i * step;
        A[n * (n - 1) + i] = corners[3] + i * step;
    }
    std::memcpy(Anew.get(), A.get(), n * n * sizeof(double));
}

__global__ void Calculate_matrix(double *A, double *Anew, size_t size)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i * size + j >= size * size)
        return;

    if (!((j == 0 || i == 0 || j >= size - 1 || i >= size - 1)))
    {
        Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] +
                                     A[(i + 1) * size + j] + A[i * size + j + 1]);
    }
}

__global__ void Error_matrix(double *Anew, double *A, double *error, int size)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i * size + j >= size * size)
        return;

    error[i * size + j] = fabs(A[i * size + j] - Anew[i * size + j]);
}

void save(double *A, int size)
{
    std::ofstream out("out.txt");
    out << std::fixed << std::setprecision(5);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            out << std::left << std::setw(10) << A[i * size + j] << " ";
        }
        out << std::endl;
    }
}

int main(int argc, char const *argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message")("err", po::value<double>()->default_value(0.000001), "error")("size", po::value<int>()->default_value(20), "size")("iter", po::value<int>()->default_value(1000000), "number of iterations");

    po::variables_map vm;

    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }
    po::notify(vm);

    double err = vm["err"].as<double>();
    int n = vm["size"].as<int>();
    int iter_max = vm["iter"].as<int>();

    std::unique_ptr<double[]> A_ptr(new double[n * n]);
    std::unique_ptr<double[]> Anew_ptr(new double[n * n]);
    initialize(std::ref(A_ptr), std::ref(Anew_ptr), n);
    std::unique_ptr<double[]> E_ptr(new double[n * n]);

    double *A = A_ptr.get();
    double *Anew = Anew_ptr.get();
    double *error_matrix = E_ptr.get();

    cuda_unique_ptr<cudaStream_t> stream_ptr(cuda_new_stream(), cuda_delete_stream);
    cuda_unique_ptr<cudaGraph_t> graph(cuda_new_graph(), cuda_delete_graph);
    cuda_unique_ptr<cudaGraphExec_t> graph_save(cuda_new_graph_save(), cuda_delete_graph_save);

    auto stream = *stream_ptr;

    cuda_unique_ptr<double> A_device_ptr(cuda_new(sizeof(double) * n * n), cuda_delete);
    cuda_unique_ptr<double> Anew_device_ptr(cuda_new(sizeof(double) * n * n), cuda_delete);
    cuda_unique_ptr<double> error_device_ptr(cuda_new(sizeof(double) * n * n), cuda_delete);
    cuda_unique_ptr<double> error_GPU_ptr(cuda_new(sizeof(double)), cuda_delete);

    double *error_GPU = error_GPU_ptr.get();
    double *error_device = error_device_ptr.get();
    double *Anew_device = Anew_device_ptr.get();
    double *A_device = A_device_ptr.get();

    cudaError_t cudaErr1 = cudaMemcpy(A_device, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaError_t cudaErr2 = cudaMemcpy(Anew_device, Anew, n * n * sizeof(double), cudaMemcpyHostToDevice);

    if (cudaErr1 != cudaSuccess || cudaErr2 != cudaSuccess)
    {
        std::cerr << "Memory transfering error: " << cudaGetErrorString(cudaErr1) << ", " << cudaGetErrorString(cudaErr2) << std::endl;
        exit(1);
    }

    cuda_unique_ptr<double> tmp_ptr_old(cuda_new(0), cuda_delete);
    double *tmp_old = tmp_ptr_old.get();
    size_t tmp_size_old = 0;

    cub::DeviceReduce::Max(tmp_old, tmp_size_old, Anew_device, error_GPU, n * n);

    size_t tmp_size = tmp_size_old;
    cuda_unique_ptr<double> tmp_ptr(cuda_new(tmp_size), cuda_delete);
    double *tmp = tmp_ptr.get();

    dim3 block = dim3(32, 32);
    dim3 grid(ceil(n / block.x), ceil(n / block.y));

    int calc_err = 0;

    double error = 1.0;
    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (error > err && iter < iter_max)
    {
        if (!calc_err)
        {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            for (size_t i = 0; i < 100; i++)
            {
                Calculate_matrix<<<grid, block, 0, stream>>>(Anew_device, A_device, n);

                double *temp = A_device;
                A_device = Anew_device;
                Anew_device = temp;
            }

            Error_matrix<<<grid, block, 0, stream>>>(Anew_device, A_device, error_device, n);

            cudaStreamEndCapture(stream, graph.get());
            cudaGraphInstantiate(graph_save.get(), *graph, NULL, NULL, 0);

            calc_err = 1;
        }

        else
        {
            cudaGraphLaunch(*graph_save, stream);
            cub::DeviceReduce::Max(tmp, tmp_size, error_device, error_GPU, n * n, stream);
            cudaError_t cudaErr3 = cudaMemcpy(&error, error_GPU, sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaErr3 != cudaSuccess)
            {
                std::cerr << "Memory transfering error: " << cudaGetErrorString(cudaErr3) << std::endl;
                exit(1);
            }

            iter += 100;
            printf("%5d, %0.6f\n", iter, error);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - start;

    cudaError_t cudaErr4 = cudaMemcpy(A, A_device, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaErr4 != cudaSuccess)
    {
        std::cerr << "Memory transfering error: " << cudaGetErrorString(cudaErr4) << std::endl;
        exit(1);
    }

    save(A, n);

    printf(" total: %f s\n", runtime.count());

    return 0;
}
