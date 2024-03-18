#include <iostream>
#include <thread>
#include <vector>

void intialize_vector(int num_threads, std::vector<double> &vec)
{

    int chunck_size = vec.size() / num_threads;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++)
    {
        int start = i * chunck_size, end = (i == num_threads - 1) ? (i + 1) * chunck_size : vec.size() - i * chunck_size;
        threads.emplace_back([start, end, &vec]()
                             {
            for (int j = start; j < end; ++j) {
                vec[j] = 1;
            } });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
}

std::vector<double> multiply_vector_matrix(const std::vector<double> &vec, const std::vector<double> &matrix, int num_threads)
{
    int vec_size = vec.size();
    std::vector<double> result(vec_size);

    std::vector<std::thread> threads;
    int chunck_size = matrix.size() / num_threads;

    for (int i = 0; i < num_threads; i++)
    {
        int start = i * chunck_size, end = (i == num_threads - 1) ? (i + 1) * chunck_size : vec.size() - i * chunck_size;
        threads.emplace_back([start, end, &vec, &matrix, &result, vec_size]()
                             {
            for (int i = 0; i < vec_size * vec_size; ++i) {
                result[i / vec_size] += vec[i % vec_size] * matrix[(i / vec_size) * vec_size + i % vec_size];
    } });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    return result;
}

int main(int argc, char **argv)
{

    int N = 20000, num_threads = 1;

    if (argc > 1)
    {
        N = atoi(argv[1]);
    }

    if (argc > 2)
    {
        num_threads = atoi(argv[2]);
        
    }

    std::vector<double> vec(N);
    intialize_vector(num_threads, vec);

    std::vector<double> matrix(N * N);
    intialize_vector(num_threads, matrix);


    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<double> res = multiply_vector_matrix(vec, matrix, num_threads);
    std::chrono::duration<double, std::milli> execution_time = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << (double)execution_time.count() / 1000;
}