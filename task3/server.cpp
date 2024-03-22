#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <fstream>


size_t N = 100;

template<typename T>
class Task {
private:
    std::function<T(T, T)> _function;

public:
    size_t x, y;
    Task(size_t _x, size_t _y, std::function<T(T, T)> function) : x(_x), y(_y), _function(function) {}

    T execute_function() {
        return _function(x, y);
    }

    std::function<T()> get_task_function() {
        return [this]() { return this->execute_function(); };
    }
};


template<typename T>
class server {
private:
    std::queue<std::pair<size_t, std::packaged_task<T()>>> tasks;
    std::unordered_map<size_t, T> results;
    std::mutex mut;
    std::thread serv_thr;
    bool server_is_up = false;
    size_t available_id = 0;

    void server_thread() {
        std::unique_lock<std::mutex> lock_res{mut, std::defer_lock};
        size_t id_task;

        while (server_is_up) {
            lock_res.lock();

            if (!tasks.empty()) {
                auto task = std::move(tasks.front());
                tasks.pop();
                std::future<T> fut = task.second.get_future();
                task.second();
                T result = fut.get();
                results.insert({task.first, result});
            }
            lock_res.unlock();
            // std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

public:
    void start() {
        if (!server_is_up) {
            server_is_up = true;
            serv_thr = std::thread (&server::server_thread, this);
        }
    }

    void stop() {
        server_is_up = false;
        serv_thr.join();
    }

    size_t add_task_thread(Task<T> t) {
    std::unique_lock<std::mutex> lock(mut);

    std::function<T()> func = t.get_task_function();
    std::packaged_task<T()> pt(std::move(func));
    tasks.push({available_id, std::move(pt)});
    
    size_t id = available_id++;
    
    return id;
}


    T request_result(size_t id){
        auto it = results.find(id);
        if (it != results.end()) {
            return results[id];
        }
        return -1;
    }
};

template <typename T>
T fun_pow(T x, T y) {
    return std::pow(x, y);
}

template <typename T>
T fun_sin(T x, T y) {
    return std::sin(x);
}

template <typename T>
T fun_sqrt(T x, T y) {
    return std::sqrt(x);
}



template<typename T, typename Func>
void run_client(server<T> &s, int N, Func func, std::string file_name) {
    std::ofstream file(file_name);
    for (int i = 0; i < N; i++) {
        Task<T> task(i, i, func);
        size_t id = s.add_task_thread(task);
        T req = s.request_result(id);
        for (; req == -1; req = s.request_result(id))
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        file << "ID: " << id << "\nresult: " << req << std::endl;
    }
}


int main() {
    server<double> s;
    s.start();
    std::cout << "Server started..." << std::endl;


    std::thread client1([&s]() { run_client<double>(s, N, fun_pow<double>, "pow.txt");});
    std::thread client2([&s]() { run_client<double>(s, N, fun_sin<double>, "sin.txt");});
    std::thread client3([&s]() { run_client<double>(s, N, fun_sqrt<double>, "sqrt.txt");});

    client1.join();
    client2.join();
    client3.join();

    s.stop();
    std::cout << "Server closed..." << std::endl;

    return 0;
}
