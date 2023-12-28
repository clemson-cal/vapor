#pragma once
#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif
#include <limits>
#include "core_index_space.hpp"
#include "core_memory.hpp"

namespace vapor {





struct cpu_executor_t : public allocation_pool_t
{
    template<typename F>
    void loop(index_space_t<1> space, F function) const
    {
        uint i0 = space.i0[0];
        uint i1 = space.i0[0] + space.di[0];

        for (uint i = i0; i < i1; ++i)
        {
            function(vec(i));
        }
    }

    template<typename F>
    void loop(index_space_t<2> space, F function) const
    {
        uint i0 = space.i0[0];
        uint i1 = space.i0[0] + space.di[0];
        uint j0 = space.i0[1];
        uint j1 = space.i0[1] + space.di[1];

        for (uint i = i0; i < i1; ++i)
            for (uint j = j0; j < j1; ++j)
                function(vec(i, j));
    }

    template<typename F>
    void loop(index_space_t<3> space, F function) const
    {
        uint i0 = space.i0[0];
        uint i1 = space.i0[0] + space.di[0];
        uint j0 = space.i0[1];
        uint j1 = space.i0[1] + space.di[1];
        uint k0 = space.i0[2];
        uint k1 = space.i0[2] + space.di[2];

        for (uint i = i0; i < i1; ++i)
            for (uint j = j0; j < j1; ++j)
                for (uint k = k0; k < k1; ++k)
                    function(vec(i, j, k));
    }

    template<typename T, typename R>
    T reduce(const T* data, size_t size, R reducer, T start) const
    {
        auto result = start;

        for (size_t i = 0; i < size; ++i)
            result = reducer(result, data[i]);
        return result;
    }
};




struct omp_executor_t : public allocation_pool_t
{
    template<typename F>
    void loop(index_space_t<1> space, F function) const
    {
        #ifdef _OPENMP
        uint i0 = space.i0[0];
        uint i1 = space.i0[0] + space.di[0];

        #pragma omp parallel for
        for (uint i = i0; i < i1; ++i)
            function(uvec(i));
        #endif
    }

    template<typename F>
    void loop(index_space_t<2> space, F function) const
    {
        #ifdef _OPENMP
        uint i0 = space.i0[0];
        uint i1 = space.i0[0] + space.di[0];
        uint j0 = space.i0[1];
        uint j1 = space.i0[1] + space.di[1];

        #pragma omp parallel for
        for (uint i = i0; i < i1; ++i)
            for (uint j = j0; j < j1; ++j)
                function(uvec(i, j));
        #endif
    }

    template<typename F>
    void loop(index_space_t<3> space, F function) const
    {
        #ifdef _OPENMP
        uint i0 = space.i0[0];
        uint i1 = space.i0[0] + space.di[0];
        uint j0 = space.i0[1];
        uint j1 = space.i0[1] + space.di[1];
        uint k0 = space.i0[2];
        uint k1 = space.i0[2] + space.di[2];

        #pragma omp parallel for
        for (uint i = i0; i < i1; ++i)
            for (uint j = j0; j < j1; ++j)
                for (uint k = k0; k < k1; ++k)
                    function(uvec(i, j, k));
        #endif
    }

    template<typename T, typename R>
    T reduce(const T* data, size_t size, R reducer, T start) const
    {
        auto result = start;

        for (size_t i = 0; i < size; ++i)
            result = reducer(result, data[i]);
        return result;
    }
};









#ifdef __CUDACC__

template<typename F, uint D>
__global__ static void gpu_loop(index_space_t<D> space, F function)
{
    if constexpr (D == 1)
    {
        uint i = space.i0[0] + threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= space.i0[0] + space.di[0]) return;
        function(uvec(i));
    }

    if constexpr (D == 2)
    {
        uint i = space.i0[0] + threadIdx.x + blockIdx.x * blockDim.x;
        uint j = space.i0[1] + threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= space.i0[0] + space.di[0]) return;
        if (j >= space.i0[1] + space.di[1]) return;
        function(uvec(i, j));
    }

    if constexpr (D == 3)
    {
        uint i = space.i0[0] + threadIdx.x + blockIdx.x * blockDim.x;
        uint j = space.i0[1] + threadIdx.y + blockIdx.y * blockDim.y;
        uint k = space.i0[2] + threadIdx.z + blockIdx.z * blockDim.z;
        if (i >= space.i0[0] + space.di[0]) return;
        if (j >= space.i0[1] + space.di[1]) return;
        if (k >= space.i0[2] + space.di[2]) return;
        function(uvec(i, j, k));
    }
}




static const dim3 THREAD_BLOCK_SIZE_1D(64, 1, 1);
static const dim3 THREAD_BLOCK_SIZE_2D(8, 8, 1);
static const dim3 THREAD_BLOCK_SIZE_3D(4, 4, 4);




struct gpu_executor_t : public allocation_pool_t
{
    template<typename F>
    void loop(index_space_t<1> space, F function) const
    {
        int device = 0;
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        space.decompose(num_devices, [&device, function] (auto subspace)
        {
            cudaSetDevice(device);
            auto ni = subspace.di[0];
            auto bs = THREAD_BLOCK_SIZE_1D;
            auto nb = dim3((ni + bs.x - 1) / bs.x);
            gpu_loop<<<nb, bs>>>(subspace, function);
            device += 1;
        });
    }

    template<typename F>
    void loop(index_space_t<2> space, F function) const
    {
        int device = 0;
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        space.decompose(num_devices, [&device, function] (auto subspace)
        {
            auto ni = subspace.di[0];
            auto nj = subspace.di[1];
            auto bs = THREAD_BLOCK_SIZE_2D;
            auto nb = dim3((ni + bs.x - 1) / bs.x, (nj + bs.y - 1) / bs.y);
            gpu_loop<<<nb, bs>>>(subspace, function);
        });
    }

    template<typename F>
    void loop(index_space_t<3> space, F function) const
    {
        int device = 0;
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        space.decompose(num_devices, [&device, function] (auto subspace)
        {
            auto ni = subspace.di[0];
            auto nj = subspace.di[1];
            auto nk = subspace.di[2];
            auto bs = THREAD_BLOCK_SIZE_3D;
            auto nb = dim3((ni + bs.x - 1) / bs.x, (nj + bs.y - 1) / bs.y, (nk + bs.z - 1) / bs.z);
            gpu_loop<<<nb, bs>>>(subspace, function);
        });
    }

    template<typename T, typename R>
    T reduce(const T* data, size_t size, R reducer, T start) const
    {
        T result;
        T *result_buf = nullptr;
        void *scratch = nullptr;
        size_t scratch_bytes = 0;
        cub::DeviceReduce::Reduce(scratch, scratch_bytes, data, result_buf, size, reducer, start);
        cudaMalloc(&result_buf, sizeof(T));
        cudaMalloc(&scratch, scratch_bytes);
        cub::DeviceReduce::Reduce(scratch, scratch_bytes, data, result_buf, size, reducer, start);
        cudaMemcpy(&result, result_buf, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(result_buf);
        cudaFree(scratch);
        return result;
    }
};
#endif // __CUDACC__




#ifdef __CUDACC__
using default_executor_t = gpu_executor_t;
#else
using default_executor_t = cpu_executor_t;
#endif

} // namespace vapor
