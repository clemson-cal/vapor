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
Rationale:

This library relies heavily on CUDA's unified memory model. Blocks of memory
to be managed by the CUDA runtime have a move-only RAII container called
buffer_t. On non-CUDA platforms the allocation reduces to malloc / free.

Reference counted, managed memory allocations are used as the backing buffers
for cached arrays. An allocator pool maintains a list of buffer_t instances
and corresponding use counts, and vends out unused allocations as e.g.
ref_counted_ptr_t<double>. The ref-counted container automatically increments
and decrements the use counts. Vended allocations serve as the backing
buffers for cached arrays. An allocation with a zero use count is available
to be vended out again by the pool.

The pool must outlive any arrays that were cached using the pool's
allocations. It is an error to return a cached array from a function, if the
allocator pool was also created in the function. If a function needs to return
a cached array, it should also take the allocator pool as an argument.
================================================================================
*/




#pragma once
#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <memory>
#include "compat.hpp"

namespace vapor {




/**
 * An RAII read-write block of host, device, or managed memory
 *
 * On non-CUDA platforms the memory is ordinary CPU memory
 * 
 */
class buffer_t
{
public:
    buffer_t(const buffer_t& other) = delete;
    buffer_t(buffer_t&& other)
    {
        release();
        _data = other._data;
        _bytes = other._bytes;
        _device = other._device;
        other._data = nullptr;
        other._bytes = 0;
        other._device = -1;
    }
    buffer_t()
    {
    }
    buffer_t(size_t bytes)
    {
        allocate(bytes);
    }
    ~buffer_t()
    {
        release();
    }
    bool managed() const
    {
        return _device == -1;
    }
    void allocate(size_t bytes, int device=-1)
    {
        if (bytes > _bytes || device != _device) {
            release();
            _bytes = bytes;
            _device = device;
            #ifdef __CUDACC__
            if (device == -1) {
                cudaMallocManaged(&_data, _bytes);
            }
            else {
                cudaSetDevice(_device);
                cudaMalloc(&_data, _bytes);
            }
            #else
            _data = malloc(_bytes);
            #endif
        }
    }
    int device() const
    {
        return _device;
    }
    size_t bytes() const
    {
        return _bytes;
    }
    template <typename T> size_t size() const
    {
        return _bytes / sizeof(T);
    }
    template <typename T> auto *data() const
    {
        return (const T*)_data;
    }
    template <typename T> auto *data()
    {
        return (T*)_data;
    }
    template <typename T> T read(size_t i) const
    {
        #ifdef __CUDACC__
        if (_device == -1) {
            return ((T*)_data)[i];
        }
        else {
            T value;
            cudaSetDevice(_device);
            cudaMemcpy(&value, (T*)_data + i, sizeof(T), cudaMemcpyDeviceToHost);
            return value;
        }
        #else
        return ((T*)_data)[i];
        #endif
    }
    template <typename T> void write(size_t i, const T& value) const
    {
        #ifdef __CUDACC__
        if (_device == -1) {
            ((T*)_data)[i] = value;
        }
        else {
            cudaSetDevice(_device);
            cudaMemcpy((T*)_data + i, &value, sizeof(T), cudaMemcpyHostToDevice);
        }
        #else
        ((T*)_data)[i] = value;
        #endif
    }
private:
    void release()
    {
        #ifdef __CUDACC__
        if (_device != -1) {
            cudaSetDevice(_device);
        }
        cudaFree(_data);
        #else
        free(_data);
        #endif
        _data = nullptr;
        _bytes = 0;
    }
    void *_data = nullptr;
    size_t _bytes = 0;
    int _device = -1;
};




/**
 * A minimal unique pointer to a single POD item in managed memory
 *
 * This class is not currently in use by the library and may be removed in the
 * future.
 */
template<typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
class managed_memory_ptr_t
{
public:
    managed_memory_ptr_t(const managed_memory_ptr_t& other) = delete;
    managed_memory_ptr_t(managed_memory_ptr_t&& other) = default;
    managed_memory_ptr_t(T val)
    {
        mem.allocate(sizeof(T));
        _data = mem.template data<T>();
        _data[0] = val;
    }
    const T* get() const
    {
        return _data;
    }
    T* get()
    {
        return _data;
    }
    const T& operator*() const
    {
        return *_data;
    }
    T& operator*()
    {
        return *_data;
    }
private:
    buffer_t mem;
    T *_data;
};




/**
 * A non-deallocating, non-thread-safe, reference counted pointer type
 *
 * This container does not automatically release any resources, it merely
 * adds a use count to an otherwise raw pointer value.
 *
 * Reference counting is disabled in device code, since each GPU thread would
 * otherwise try to upate the use count. It is consistent with required
 * behavior, since arrays are only allocated or freed in host code anyway.
 */
template<class T>
class ref_counted_ptr_t
{
public:
    ref_counted_ptr_t() {}
    ref_counted_ptr_t(T *ptr, int *use_count) : _ptr(ptr), _use_count(use_count)
    {
        assert(*use_count == 0);
        retain();
    }
    HD ref_counted_ptr_t(const ref_counted_ptr_t& other)
    {
        _use_count = other._use_count;
        _ptr = other._ptr;
        retain();
    }
    HD ref_counted_ptr_t& operator=(const ref_counted_ptr_t& other)
    {
        release();
        _use_count = other._use_count;
        _ptr = other._ptr;
        retain();
        return *this;
    }
    HD ~ref_counted_ptr_t()
    {
        release();
    }
    HD T* get()
    {
        return _ptr;
    }
    HD const T* get() const
    {
        return _ptr;
    }
    T* operator->()
    {
        return _ptr;
    }
    const T* operator->() const
    {
        return _ptr;
    }
private:
    HD void retain()
    {
        #ifndef __CUDA_ARCH__
        if (_use_count != nullptr)
        {
            *_use_count += 1;
        }
        #endif
    }
    HD void release()
    {
        #ifndef __CUDA_ARCH__
        if (_use_count != nullptr)
        {
            *_use_count -= 1;
        }
        #endif
    }
    T *_ptr = nullptr;
    int *_use_count = nullptr;
};




/**
 * A pool of reference-counted, managed memory allocations
 *
 */
class pool_allocator_t
{
public:
    using allocation_t = ref_counted_ptr_t<buffer_t>;

    pool_allocator_t& operator=(const pool_allocator_t& other) = delete;
    pool_allocator_t(const pool_allocator_t& other) = delete;
    pool_allocator_t(pool_allocator_t&& other)
    {
        delete [] allocations;
        delete [] use_counts;
        num_allocations = other.num_allocations;
        allocations = other.allocations;
        use_counts = other.use_counts;
        other.num_allocations = 0;
        other.allocations = nullptr;
        other.use_counts = nullptr;
    }
    pool_allocator_t(uint num_allocations=1024) : num_allocations(num_allocations)
    {
        allocations = new buffer_t[num_allocations];
        use_counts = new int[num_allocations];

        for (uint n = 0; n < num_allocations; ++n)
        {
            use_counts[n] = 0;
        }
    }
    allocation_t allocate(size_t bytes, int device=-1) const
    {
        for (size_t n = 0; n < num_allocations; ++n)
        {
            if (use_counts[n] == 0 && (allocations[n].bytes() == 0 || allocations[n].device() == device))
            {
                allocations[n].allocate(bytes, device);
                return ref_counted_ptr_t<buffer_t>(&allocations[n], &use_counts[n]);
            }
        }
        assert(false); // the pool is out of allocations
    }
    ~pool_allocator_t()
    {
        for (size_t n = 0; n < num_allocations; ++n)
        {
            assert(use_counts[n] == 0); // an allocation is trying to outlive the pool
        }
        delete [] allocations;
        delete [] use_counts;
    }
private:
    size_t num_allocations = 0;
    buffer_t* allocations = nullptr;
    int* use_counts = nullptr;
};




/**
 * Allocator returning a std::shared_ptr to a block of managed memory
 *
 * The allocations returned are thread-safe, and can safely outlive this
 * allocator, however they cannot be used in device code, and should not be
 * used in performance-critical code, as repeated calls to system malloc/free
 * will be incurred than with the pool allocator.
 *
 */
class shared_ptr_allocator_t
{
public:
    using allocation_t = std::shared_ptr<buffer_t>;

    allocation_t allocate(size_t bytes, int device=-1) const
    {
        return std::make_shared<buffer_t>(bytes);
    }
};




#ifdef VAPOR_USE_SHARED_PTR_ALLOCATOR
using default_allocator_t = shared_ptr_allocator_t;
#else
using default_allocator_t = pool_allocator_t;
#endif

} // namespace vapor
