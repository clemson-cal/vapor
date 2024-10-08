/**
================================================================================
Copyright 2023 - 2024, Jonathan Zrake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
================================================================================




Executors provide four functions:

loop:       (index_space, function) -> future<void>
loop_async: (index_space, function, device) -> none
loop_accum: (index_space, function, allocator) -> future<int>
reduce:     (buffer, reducer: (T, T) -> T, start: T, allocator) -> future<T>

The reason that loop_async returns nothing, is that the buffer is either ready
immediately (CPU executor), or it was allocated on a device, and any
subsequent kernel launches to the same device would be ordered by the CUDA
runtime.
*/




#pragma once
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif
#include "index_space.hpp"
#include "memory.hpp"
#include "future.hpp"

namespace vapor {




struct cpu_executor_t
{
    template <typename T, class A>
    using reduce_future_t = future::future_t<future::just_t<T>>;

    template<typename F>
    auto loop(index_space_t<1> space, F function) const
    {
        int i0 = space.i0[0];
        int i1 = space.i0[0] + space.di[0];

        for (int i = i0; i < i1; ++i)
            function(vec(i));
        return future::ready();
    }

    template<typename F>
    auto loop(index_space_t<2> space, F function) const
    {
        int i0 = space.i0[0];
        int i1 = space.i0[0] + space.di[0];
        int j0 = space.i0[1];
        int j1 = space.i0[1] + space.di[1];

        for (int i = i0; i < i1; ++i)
            for (int j = j0; j < j1; ++j)
                function(vec(i, j));
        return future::ready();
    }

    template<typename F>
    auto loop(index_space_t<3> space, F function) const
    {
        int i0 = space.i0[0];
        int i1 = space.i0[0] + space.di[0];
        int j0 = space.i0[1];
        int j1 = space.i0[1] + space.di[1];
        int k0 = space.i0[2];
        int k1 = space.i0[2] + space.di[2];

        for (int i = i0; i < i1; ++i)
            for (int j = j0; j < j1; ++j)
                for (int k = k0; k < k1; ++k)
                    function(vec(i, j, k));
        return future::ready();
    }

    template<uint D, typename F, class A>
    auto loop_accumulate(index_space_t<D> space, F function, A&) const
    {
        auto c = int();
        auto g = [function, &c] (ivec_t<D> i)
        {
            c += function(i);
        };
        loop(space, g);
        return future::just(c);
    }

    template<uint D, typename F>
    void loop_async(index_space_t<D> space, int device, F function) const
    {
        return loop(space, function).get();
    }

    template<typename T, class R, class A>
    auto reduce(const buffer_t& buffer, R reducer, T start, A&) const
    {
        auto data = buffer.template data<T>();
        auto size = buffer.template size<T>();
        auto result = start;
        for (size_t i = 0; i < size; ++i)
            result = reducer(result, data[i]);
        return future::just(result);
    }

    auto num_devices() const { return 1; }
};




#ifdef _OPENMP
struct omp_executor_t
{
    template <typename T, class A>
    using reduce_future_t = future::future_t<future::just_t<T>>;

    template<typename F>
    auto loop(index_space_t<1> space, F function) const
    {
        int i0 = space.i0[0];
        int i1 = space.i0[0] + space.di[0];

        #pragma omp parallel for
        for (int i = i0; i < i1; ++i)
            function(vec(i));
        return future::ready();
    }

    template<typename F>
    auto loop(index_space_t<2> space, F function) const
    {
        int i0 = space.i0[0];
        int i1 = space.i0[0] + space.di[0];
        int j0 = space.i0[1];
        int j1 = space.i0[1] + space.di[1];

        #pragma omp parallel for
        for (int i = i0; i < i1; ++i)
            for (int j = j0; j < j1; ++j)
                function(vec(i, j));
        return future::ready();
    }

    template<typename F>
    auto loop(index_space_t<3> space, F function) const
    {
        int i0 = space.i0[0];
        int i1 = space.i0[0] + space.di[0];
        int j0 = space.i0[1];
        int j1 = space.i0[1] + space.di[1];
        int k0 = space.i0[2];
        int k1 = space.i0[2] + space.di[2];

        #pragma omp parallel for
        for (int i = i0; i < i1; ++i)
            for (int j = j0; j < j1; ++j)
                for (int k = k0; k < k1; ++k)
                    function(vec(i, j, k));
        return future::ready();
    }

    template<uint D, typename F, class A>
    auto loop_accumulate(index_space_t<D> space, F function, A&) const
    {
        auto c = int();
        auto g = [function, &c] (ivec_t<D> i)
        {
            if (auto res = function(i)) {
                #pragma omp atomic update
                c += function(i);
            }
        };
        loop(space, g);
        return future::just(c);
    }

    template<uint D, typename F>
    void loop_async(index_space_t<D> space, int device, F function) const
    {
        return loop(space, function).get();
    }

    template<typename T, class R, class A>
    auto reduce(const buffer_t& buffer, R reducer, T start, A&) const
    {
        auto data = buffer.template data<T>();
        auto size = buffer.template size<T>();
        auto result = start;
        for (size_t i = 0; i < size; ++i)
            result = reducer(result, data[i]);
        return future::just(result);
    }

    auto num_devices() const { return 1; }
};
#endif // _OPENMP




#ifdef __CUDACC__
template<typename F, uint D>
__global__ static void gpu_loop(index_space_t<D> space, F function)
{
    if constexpr (D == 1)
    {
        auto i = ivec(
            space.i0[0] + threadIdx.x + blockIdx.x * blockDim.x
        );
        if (space.contains(i)) {
            function(i);
        }
    }
    if constexpr (D == 2)
    {
        auto i = ivec(
            space.i0[0] + threadIdx.x + blockIdx.x * blockDim.x,
            space.i0[1] + threadIdx.y + blockIdx.y * blockDim.y
        );
        if (space.contains(i)) {
            function(i);
        }
    }
    if constexpr (D == 3)
    {
        auto i = ivec(
            space.i0[0] + threadIdx.x + blockIdx.x * blockDim.x,
            space.i0[1] + threadIdx.y + blockIdx.y * blockDim.y,
            space.i0[2] + threadIdx.z + blockIdx.z * blockDim.z
        );
        if (space.contains(i)) {
            function(i);
        }
    }
}




static const dim3 THREAD_BLOCK_SIZE_1D(64, 1, 1);
static const dim3 THREAD_BLOCK_SIZE_2D(8, 8, 1);
static const dim3 THREAD_BLOCK_SIZE_3D(4, 4, 4);




template<typename T, class A>
struct read_from_buffer_t
{
    using BufferHolder = typename A::allocation_t;
    auto operator()() const
    {
        return buffer_holder->template read<T>(0);
    }
    BufferHolder buffer_holder;
};

struct device_synchronize_t
{
    void operator()() const
    {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }
    int device;
};

template<uint D, typename F>
struct loop_accumulate_t
{
    HD void operator()(ivec_t<D> i)
    {
        atomicAdd(c_ptr, function(i));
    }
    F function;
    int *c_ptr;
};




struct gpu_executor_t
{
    template <typename T, class A>
    using reduce_future_t = future::future_t<read_from_buffer_t<T, A>>;

    template <class A>
    using loop_accumulate_future_t = future::future_t<read_from_buffer_t<int , A>>;

    gpu_executor_t(int num_devices=-1)
    {
        int num_devices_available;
        cudaGetDeviceCount(&num_devices_available);

        if (num_devices != -1) {
            _num_devices = num_devices;
        }
        else if (auto str = std::getenv("VAPOR_NUM_DEVICES")) {
            _num_devices = atoi(str);
        }
        else {
            _num_devices = num_devices_available;
        }
        if (_num_devices > num_devices_available) {
            throw std::runtime_error("VAPOR_NUM_DEVICES is greater than the number of devices");
        }
        if (_num_devices <= 0) {
            throw std::runtime_error("VAPOR_NUM_DEVICES must be greater than zero");
        }
        if (_num_devices > VAPOR_MAX_DEVICES) {
            throw std::runtime_error("VAPOR_MAX_DEVICES needs to be increased");
        }
    }

    template<uint D, typename F>
    auto loop(index_space_t<D> space, F function) const
    {
        for (int device = 0; device < _num_devices; ++device)
        {
            auto subspace = space.subspace(_num_devices, device);
            if (! subspace.empty()) {
                loop_async(subspace, device, function);
            }
        }
        for (int device = 0; device < _num_devices; ++device)
        {
            cudaSetDevice(device);
            cudaDeviceSynchronize();
        }
        return future::ready();
    }

    template<typename F>
    auto loop_async(index_space_t<1> space, int device, F function) const
    {
        cudaSetDevice(device);
        auto ni = space.di[0];
        auto bs = THREAD_BLOCK_SIZE_1D;
        auto nb = dim3((ni + bs.x - 1) / bs.x);
        gpu_loop<<<nb, bs>>>(space, function);
        return future::future(device_synchronize_t{device});
    }

    template<typename F>
    auto loop_async(index_space_t<2> space, int device, F function) const
    {
        cudaSetDevice(device);
        auto ni = space.di[0];
        auto nj = space.di[1];
        auto bs = THREAD_BLOCK_SIZE_2D;
        auto nb = dim3((ni + bs.x - 1) / bs.x, (nj + bs.y - 1) / bs.y);
        gpu_loop<<<nb, bs>>>(space, function);
        return future::future(device_synchronize_t{device});
    }

    template<typename F>
    auto loop_async(index_space_t<3> space, int device, F function) const
    {
        cudaSetDevice(device);
        auto ni = space.di[0];
        auto nj = space.di[1];
        auto nk = space.di[2];
        auto bs = THREAD_BLOCK_SIZE_3D;
        auto nb = dim3((ni + bs.x - 1) / bs.x, (nj + bs.y - 1) / bs.y, (nk + bs.z - 1) / bs.z);
        gpu_loop<<<nb, bs>>>(space, function);
        return future::future(device_synchronize_t{device});
    }

    template<uint D, typename F, class A>
    loop_accumulate_future_t<A> loop_accumulate(index_space_t<D> space, F function, A& allocator) const
    {
        auto c_buf = allocator.allocate(sizeof(int));
        auto c_ptr = c_buf->template data<int>();
        *c_ptr = 0;
        auto g = [function, c_ptr] HD (ivec_t<D> i)
        {
            if (auto res = function(i)) {
                atomicAdd(c_ptr, res);
            }
        };
        loop(space, g);
        return future::future(read_from_buffer_t<int, A>{c_buf});
    }

    /**
     * This reduce operator returns a buffer, immediately
     * 
     * The buffer must be a device allocation, i.e. it cannot be managed.
     * Reading from the buffer via buffer_t::read will block until the
     * reduction is completed.
     */
    template<typename T, class R, class A>
    auto reduce(const buffer_t& buffer, R reducer, T start, A& allocator) const
    {
        assert(! buffer.managed());
        cudaSetDevice(buffer.device());
        auto scratch_bytes = size_t(0);
        auto data = buffer.template data<T>();
        auto size = buffer.template size<T>();
        cub::DeviceReduce::Reduce(nullptr, scratch_bytes, data, (T*)nullptr, size, reducer, start);
        auto scratch = allocator.allocate(scratch_bytes, buffer.device());
        auto results = allocator.allocate(sizeof(T), buffer.device());
        cub::DeviceReduce::Reduce(
            scratch->template data<T>(),
            scratch_bytes,
            data,
            results->template data<T>(),
            size,
            reducer,
            start
        );
        return future::future(read_from_buffer_t<T, A>{results});
    }

    auto num_devices() const { return _num_devices; }

    int _num_devices;
};
#endif // __CUDACC__




#if defined(__CUDACC__)
using default_executor_t = gpu_executor_t;
#elif defined(_OPENMP)
using default_executor_t = omp_executor_t;
#else
using default_executor_t = cpu_executor_t;
#endif

} // namespace vapor
