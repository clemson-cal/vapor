/**
 * Illustrates a possible implementation of a a thread pool executor.
 *
 * This example is neat, but it's not performant and has questionable utility
 * given the generally better performance with OpenMP and the omp_executor_t.
 *
 * Also the thread pool executor demonstrated here is (probably) not
 * compatible with a pool allocator, since VAPOR's reference counted pointer
 * is not thread-safe.
 * 
 */
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include "app_print.hpp"
#include "core_array.hpp"
#include "core_executor.hpp"
#include "core_memory.hpp"




class worker_t
{
public:
    worker_t() : t([this] {
        while (true) {
            auto lock = std::lock_guard<std::mutex>(m);
            if (f != nullptr) {
                f();
                f = nullptr;
            }
            if (stop) {
                return;
            }
        }
    }) {
    }
    ~worker_t() {
        stop = true;
        t.join();
    }
    void submit(std::function<void(void)> new_func) {
        auto lock = std::lock_guard<std::mutex>(m);
        f = new_func;
    }
    void wait() {
        while (true) {
            auto lock = std::lock_guard<std::mutex>(m);
            if (f == nullptr) {
                return;
            }
        }
    }
private:
    std::mutex m;
    std::thread t;
    std::function<void(void)> f;
    std::atomic<bool> stop = false;
};




class thread_pool_executor_t
{
public:
    thread_pool_executor_t(int num_workers)
    {
        for (int i = 0; i < num_workers; ++i)
        {
            workers.push_back(std::make_shared<worker_t>());
        }
    }
    template<vapor::uint D, typename F>
    void loop(vapor::index_space_t<D> space, F f) const
    {
        auto n = 0;
        space.decompose(workers.size(), [&n, f, this] (auto subspace)
        {
            auto g = [subspace, f, this] { base_executor.loop(subspace, f); };
            workers[n]->submit(g);
            ++n;
        });
        for (auto& w : workers)
        {
            w->wait();
        }
    }
private:
    vapor::cpu_executor_t base_executor;
    std::vector<std::shared_ptr<worker_t>> workers;
};




int main()
{
    auto exec = thread_pool_executor_t(28);
    auto alloc = vapor::shared_ptr_allocator_t();
    auto a = vapor::range(56)
        .map([] (auto i) { return vapor::dvec(i, i, i); })
        .cache(exec, alloc);

    for (int i = 0; i < 56; ++i)
    {
        vapor::print(a[i]);
        vapor::print("\n");
    }
    return 0;
}
