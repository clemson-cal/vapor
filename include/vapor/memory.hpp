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
managed_memory_t. On non-CUDA platforms the allocation reduces to malloc /
free.

Reference counted, managed memory allocations are used as the backing buffers
for cached arrays. An allocator pool maintains a list of managed_memory_t
instances and corresponding use counts, and vends out unused allocations as
e.g. ref_counted_ptr_t<double>. The ref-counted container automatically
increments and decrements the use counts. Vended allocations serve as the
backing buffers for cached arrays. An allocation with a zero use count is
available to be vended out again by the pool.

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
 * An RAII read-write block of managed device memory
 *
 * On non-CUDA platforms the memory is ordinary CPU memory
 * 
 */
class managed_memory_t
{
public:
    managed_memory_t(const managed_memory_t& other) = delete;
    managed_memory_t(managed_memory_t&& other)
    {
        release();
        _data = other._data;
        _bytes = other._bytes;
        other._data = nullptr;
        other._bytes = 0;
    }
    managed_memory_t()
    {
    }
    managed_memory_t(size_t bytes)
    {
        allocate(bytes);
    }
    ~managed_memory_t()
    {
        release();
    }
    void allocate(size_t bytes)
    {
        if (bytes > _bytes) {
            release();
            _bytes = bytes;
            #ifdef __CUDACC__
            cudaMallocManaged(&_data, _bytes);
            #else
            _data = malloc(_bytes);
            #endif
        }
    }
    void *data()
    {
        return _data;
    }
private:
    void release()
    {
        #ifdef __CUDACC__
        cudaFree(_data);
        #else
        free(_data);
        #endif
        _data = nullptr;
        _bytes = 0;
    }
    void *_data = nullptr;
    size_t _bytes = 0;
};




/**
 * A minimal unique pointer to a single POD item in managed memory
 *
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
        _data = (T*) mem.data();
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
    managed_memory_t mem;
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
        allocations = new managed_memory_t[num_allocations];
        use_counts = new int[num_allocations];

        for (uint n = 0; n < num_allocations; ++n)
        {
            use_counts[n] = 0;
        }
    }
    ref_counted_ptr_t<managed_memory_t> allocate(size_t bytes) const
    {
        for (size_t n = 0; n < num_allocations; ++n)
        {
            if (use_counts[n] == 0)
            {
                allocations[n].allocate(bytes);
                return ref_counted_ptr_t<managed_memory_t>(&allocations[n], &use_counts[n]);
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
    managed_memory_t* allocations = nullptr;
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
    std::shared_ptr<managed_memory_t> allocate(size_t count) const
    {
        return std::make_shared<managed_memory_t>(count);
    }
};




#ifdef VAPOR_USE_SHARED_PTR_ALLOCATOR
using default_allocator_t = shared_ptr_allocator_t;
#else
using default_allocator_t = pool_allocator_t;
#endif

} // namespace vapor
