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



================================================================================
Executors provide three functions:

loop:       (index_space, function) -> future<void>
loop_accum: (index_space, function, allocator) -> future<int>
reduce:     (buffer, reducer: (T, T) -> T, start: T, allocator) -> future<T>
================================================================================
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
};




#ifdef _OPENMP
struct omp_executor_t
{
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
            #pragma omp atomic update
            c += function(i);
        };
        loop(space, g);
        return future::just(c);
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
};
#endif // _OPENMP




#ifdef __CUDACC__
template<typename F, uint D>
__global__ static void gpu_loop(index_space_t<D> space, F function)
{
    if constexpr (D == 1)
    {
        int i = space.i0[0] + threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= space.i0[0] + space.di[0]) return;
        function(vec(i));
    }

    if constexpr (D == 2)
    {
        int i = space.i0[0] + threadIdx.x + blockIdx.x * blockDim.x;
        int j = space.i0[1] + threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= space.i0[0] + space.di[0]) return;
        if (j >= space.i0[1] + space.di[1]) return;
        function(vec(i, j));
    }

    if constexpr (D == 3)
    {
        int i = space.i0[0] + threadIdx.x + blockIdx.x * blockDim.x;
        int j = space.i0[1] + threadIdx.y + blockIdx.y * blockDim.y;
        int k = space.i0[2] + threadIdx.z + blockIdx.z * blockDim.z;
        if (i >= space.i0[0] + space.di[0]) return;
        if (j >= space.i0[1] + space.di[1]) return;
        if (k >= space.i0[2] + space.di[2]) return;
        function(vec(i, j, k));
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
    __device__ void operator()(ivec_t<D> i)
    {
        atomicAdd(c_ptr, function(i));
    }
    int *c_ptr;
    F function;
};




struct gpu_executor_t
{
    gpu_executor_t(int device=0) : _device(device) { }

    template<typename F>
    auto loop(index_space_t<1> space, F function) const
    {
        cudaSetDevice(_device);
        auto ni = space.di[0];
        auto bs = THREAD_BLOCK_SIZE_1D;
        auto nb = dim3((ni + bs.x - 1) / bs.x);
        gpu_loop<<<nb, bs>>>(space, function);
        return future::future(device_synchronize_t{_device});
    }

    template<typename F>
    auto loop(index_space_t<2> space, F function) const
    {
        cudaSetDevice(_device);
        auto ni = space.di[0];
        auto nj = space.di[1];
        auto bs = THREAD_BLOCK_SIZE_2D;
        auto nb = dim3((ni + bs.x - 1) / bs.x, (nj + bs.y - 1) / bs.y);
        gpu_loop<<<nb, bs>>>(space, function);
        return future::future(device_synchronize_t{_device});
    }

    template<typename F>
    auto loop(index_space_t<3> space, F function) const
    {
        cudaSetDevice(_device);
        auto ni = space.di[0];
        auto nj = space.di[1];
        auto nk = space.di[2];
        auto bs = THREAD_BLOCK_SIZE_3D;
        auto nb = dim3((ni + bs.x - 1) / bs.x, (nj + bs.y - 1) / bs.y, (nk + bs.z - 1) / bs.z);
        gpu_loop<<<nb, bs>>>(space, function);
        return future::future(device_synchronize_t{_device});
    }

    template<uint D, typename F, class A>
    auto loop_accumulate(index_space_t<D> space, F function, A& allocator) const
    {
        auto c_buf = allocator.allocate(sizeof(int));
        auto c_ptr = c_buf->template data<int>();
        return loop(space, loop_accumulate_t<D, F>{c_ptr, function}).then(read_from_buffer_t<int, A>{c_buf});
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
    int _device;
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
