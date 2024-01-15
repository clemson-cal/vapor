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
*/




#pragma once
#include <cstdlib>
#include <cassert>
#include <exception>
#include <limits>
#include "compat.hpp"
#include "functional.hpp"
#include "index_space.hpp"
#include "memory.hpp"
#include "vec.hpp"
#include "runtime.hpp"

namespace vapor {




template<uint D, class F> struct array_t;
template<uint D, class F> struct array_selection_t;
template<uint D, typename T> auto uniform(T val, index_space_t<D> space);
template<uint D, typename T, template<typename> typename P>
using memory_backed_array_t = array_t<D, lookup_t<D, T, P<buffer_t>>>;




/**
 * Execute an array using the given executor and allocator
 *
 */
template<uint D, class F, class E, class A,
    class T = typename array_t<D, F>::value_type,
    class R = typename A::allocation_t>
array_t<D, lookup_t<D, T, R>> cache(const array_t<D, F>& a, E& executor, A& allocator)
{
    auto buffer = allocator.allocate(a.size() * sizeof(T));
    auto data = buffer->template data<T>();
    auto start = a.start();
    auto stride = strides_row_major(a.shape());
    auto table = lookup(start, stride, data, buffer);
    executor.loop(a.space(), [start, stride, data, a] HD (ivec_t<D> i)
    {
        data[dot(stride, i - start)] = a[i];
    });
    return array(table, a.space(), buffer.get());
}

template<uint D, class F>
auto cache(const array_t<D, F>& a)
{
    return cache(a, Runtime::executor(), Runtime::allocator());
}




/**
 * Execute an array using the given executor and allocator
 *
 */
template<uint D, class F, class E, class A,
    class T = typename array_t<D, F>::value_type,
    class R = typename A::allocation_t>
array_t<D, lookup_t<D, T, R>> cache_async(const array_t<D, F>& a, int device, E& executor, A& allocator)
{
    auto buffer = allocator.allocate(a.size() * sizeof(T), device);
    auto data = buffer->template data<T>();
    auto start = a.start();
    auto stride = strides_row_major(a.shape());
    auto table = lookup(start, stride, data, buffer);
    executor.loop_async(a.space(), device, [start, stride, data, a] HD (ivec_t<D> i)
    {
        data[dot(stride, i - start)] = a[i];
    });
    return array(table, a.space(), buffer.get());
}

template<uint D, class F>
auto cache_async(const array_t<D, F>& a, int device)
{
    return cache_async(a, device, Runtime::executor(), Runtime::allocator());
}




/**
 * An exception class indicating that an array of optionals contained a none
 * 
 */
class cache_unwrap_exception : public std::exception
{
public:
    cache_unwrap_exception(int num_failures) : _num_failures(num_failures)
    {
    }
    const char* what() const throw()
    {
        return "cache_unwrap encountered none array elements";
    }
    int num_failures() const
    {
        return _num_failures;
    }
private:
    int _num_failures;
};




/**
 * Cache and unwrap the values in an array of optional value type
 *
 * If any of the elements are none, this function throws
 * cache_unwrap_exception, which will indicate the number of none elements.
 */
template<uint D, class F, class E, class A,
    class T = typename array_t<D, F>::value_type::value_type,
    class R = typename A::allocation_t>
array_t<D, lookup_t<D, T, R>> cache_unwrap(const array_t<D, F>& a, E& executor, A& allocator)
{
    auto failure_count_buf = allocator.allocate(sizeof(int));
    auto failure_count_ptr = failure_count_buf->template data<int>();
    auto buffer = allocator.allocate(a.size() * sizeof(T));
    auto data = buffer->template data<T>();
    auto start = a.start();
    auto stride = strides_row_major(a.shape());
    auto table = lookup(start, stride, data, buffer);

    if constexpr (is_device_executor<E>::value) {
        #ifdef __CUDACC__
        executor.loop(a.space(), [start, stride, data, a, failure_count_ptr] __device__ (ivec_t<D> i)
        {
            auto maybe = a[i];
            if (maybe.has_value()) {
                data[dot(stride, i - start)] = maybe.get();
            }
            else {
                E::atomic_add(failure_count_ptr, 1);
            }
        });
        #endif
    }
    else {
        executor.loop(a.space(), [start, stride, data, a, failure_count_ptr] (ivec_t<D> i)
        {
            auto maybe = a[i];
            if (maybe.has_value()) {
                data[dot(stride, i - start)] = maybe.get();
            }
            else {
                E::atomic_add(failure_count_ptr, 1);
            }
        });
    }
    if (*failure_count_ptr > 0) {
        throw cache_unwrap_exception(*failure_count_ptr);
    }
    return array(table, a.space(), buffer.get());
}

template<uint D, class F>
auto cache_unwrap(const array_t<D, F>& a)
{
    return cache_unwrap(a,  Runtime::executor(), Runtime::allocator());
}




/**
 * A functional n-dimensional array
 *
 * An array is a D-dimensional index space, and a function f: ivec_t<D> -> T.
 * Arrays are logically immutable, a[i] returns by value an element of type T;
 * a[i] = x will not compile. Arrays are transformed mainly by mapping
 * operations. If g: T -> U then a.map(g) is an array with value type of U, and
 * the same index space as a. Array elements are computed lazily, meaning that
 * b = a.map(f).map(g).map(h) triggers the execution h(g(f(i)) each time b[i]
 * appears.
 * 
 * An array is 'cached' to a memory-backed array by calling a.cache(exec,
 * alloc), where exec is an executor and alloc is an allocator. The executor
 * can be a GPU executor on GPU-enabled platforms, or a multi-core executor
 * where OpenMP is available. The memory backed array uses strided memory
 * access to retrieve the value of a multi-dimensional index in the buffer.
 *
 * Unlike arrays from other numeric libraries, including numpy, arrays in
 * VAPOR can have a non-zero starting index. This changes the semantics of
 * inserting the values of one array into another, often for the better, and
 * is also favorable in dealing with global arrays and domain decomposition.
 * For example, if a covers the 1d index space (0, 100) and b covers (1, 99),
 * then the array resulting from a.insert(b) has the values of a at the
 * endpoints, and the values of b on the interior.
 *
 * In-place array modifications are modeled using a paradigm inspired by Jax.
 * If a is an array, the construct a = a.at(space).map(f) will map only the
 * elements inside the index space through the function f, leaving the other
 * values unchanged.
 *
 * 
 */
template<uint D, class F>
struct array_t
{
    using value_type = std::invoke_result_t<F, ivec_t<D>>;

    HD auto operator[](ivec_t<D> index) const
    {
        #ifdef VAPOR_ARRAY_BOUNDS_CHECK
        assert(space().contains(index));
        #endif
        return f(index);
    }
    HD auto operator[](int i) const
    {
        return operator[](ivec(i));
    }
    HD auto operator[](index_space_t<D> subspace) const
    {
        #ifdef VAPOR_ARRAY_BOUNDS_CHECK
        assert(space().contains(subspace));
        #endif
        return array_t<D, F>{f, subspace.di, subspace.i0};
    }
    auto start() const
    {
        return _start;
    }
    auto shape() const
    {
        return _shape;
    }
    auto space() const
    {
        return index_space(start(), shape());
    }
    auto size() const
    {
        return product(_shape);
    }
    auto data() const
    {
        return _buffer ? _buffer->template data<value_type>() : nullptr;
    }
    auto data()
    {
        return _buffer ? _buffer->template data<value_type>() : nullptr;
    }
    auto buffer() const
    {
        return _buffer;
    }
    auto at(index_space_t<D> sel) const
    {
        return array_selection_t<D, F>{sel, *this};
    }
    template<class G>
    auto insert(array_t<D, G> b)
    {
        return select(in(b.space(), space()), b, *this);
    }
	auto cache() const
    {
        return vapor::cache(*this);
	}
    auto cache_async(int device) const
    {
        return vapor::cache_async(*this, device);
    }
    auto cache_unwrap() const
    {
        return vapor::cache_unwrap(*this);
    }
    template<bool C>
    auto cache_if() const
    {
        if constexpr (C)
            return vapor::cache(*this);
        else 
            return *this;
    }

    template<class G> auto map(G g) const { return array(compose<D>(f, g), space()); }
    template<class G> auto operator+(array_t<D, G> b) const { return array(add<D>(f, b.f), space()); }
    template<class G> auto operator-(array_t<D, G> b) const { return array(sub<D>(f, b.f), space()); }
    template<class G> auto operator*(array_t<D, G> b) const { return array(mul<D>(f, b.f), space()); }
    template<class G> auto operator/(array_t<D, G> b) const { return array(div<D>(f, b.f), space()); }
    template<typename T> auto operator+(T b) const { return *this + uniform<T>(b, space()); }
    template<typename T> auto operator-(T b) const { return *this - uniform<T>(b, space()); }
    template<typename T> auto operator*(T b) const { return *this * uniform<T>(b, space()); }
    template<typename T> auto operator/(T b) const { return *this / uniform<T>(b, space()); }

    F f;
    uvec_t<D> _shape;
    ivec_t<D> _start = {0};
    buffer_t *_buffer = nullptr;
};




/**
 * An array selection is what is returned by a.at(subspace)
 *
 * The resulting object can be mapped over; the arrays
 * 
 * a.insert(a[subspace].map(f)) or a.insert(b)
 * 
 * and
 * 
 * a.at(subspace).map(f)
 *
 * are fully equivalent to one another.
 * 
 */
template<uint D, class F>
struct array_selection_t
{
    using value_type = std::invoke_result_t<F, ivec_t<D>>;

    template<class G>
    auto map(G g) { return select(in(sel, a.space()), a.map(g), a); }
    auto set(value_type v) { return this->map(constant(v)); }

    template<class G> auto operator+(array_t<D, G> b) const { return select(in(sel, a.space()), a + b, a); }
    template<class G> auto operator-(array_t<D, G> b) const { return select(in(sel, a.space()), a - b, a); }
    template<class G> auto operator*(array_t<D, G> b) const { return select(in(sel, a.space()), a * b, a); }
    template<class G> auto operator/(array_t<D, G> b) const { return select(in(sel, a.space()), a / b, a); }
    template<typename T> auto operator+(T b) const { return *this + uniform<T>(b, sel); }
    template<typename T> auto operator-(T b) const { return *this - uniform<T>(b, sel); }
    template<typename T> auto operator*(T b) const { return *this * uniform<T>(b, sel); }
    template<typename T> auto operator/(T b) const { return *this / uniform<T>(b, sel); }

    index_space_t<D> sel;
    array_t<D, F> a;
};




/**
 * A 1d array from an index function and a size
 */
template<class F>
array_t<1, F> array(F f, uint size, int start=0)
{
    return {f, uvec(size), ivec(start)};
}

/**
 * An nd array from a an index function and a shape
 */
template<uint D, class F>
array_t<D, F> array(F f, uvec_t<D> shape, ivec_t<D> start={0})
{
    return {f, shape, start};
}

/**
 * An nd array from a an index function and an index space
 */
template<uint D, class F, typename T = typename array_t<D, F>::value_type>
array_t<D, F> array(F f, index_space_t<D> space, buffer_t* buffer=nullptr)
{
    return {f, space.di, space.i0, buffer};
}

/**
 * An array of a uniform value of the given shape
 */
template<typename T, uint D>
auto uniform(T val, uvec_t<D> shape)
{
    return array(constant(val), shape);
}

template<typename T, uint D>
auto uniform(T val, index_space_t<D> space)
{
    return array(constant(val), space);
}

/**
 * An array of zeros (type T) from a shape
 */
template<typename T, uint D>
auto zeros(uvec_t<D> shape)
{
    return uniform<T, D>(T(), shape);
}

template<typename T, uint D>
auto zeros(index_space_t<D> space)
{
    return uniform<T, D>(T(), space);
}

/**
 * An array of ones (type T) from a shape
 */
template<typename T, uint D>
auto ones(uvec_t<D> shape)
{
    return uniform<T, D>(T(1), shape);
}

template<typename T, uint D>
auto ones(index_space_t<D> space)
{
    return uniform<T, D>(T(1), space);
}

/**
 * An nd array of vector-valued indices (identity array)
 */
template<uint D>
auto indices(uvec_t<D> shape)
{
    return array(identity(), shape);
}

template<uint D>
auto indices(index_space_t<D> space)
{
    return array(identity(), space);
}

/**
 * A 1d array of integer-valued indices (identity array)
 */
inline auto range(uint size)
{
    return indices(vec(size)).map(take_nth_t<ivec_t<1>>{0});
}

/**
 * 
 */
template<class E, class F, class G, uint D>
auto select(array_t<D, E> c, array_t<D, F> a, array_t<D, G> b)
{
    return array(cond<D>(c.f, a.f, b.f), c.space());
}

/**
 * A boolean array, true where the index is within the given index space
 */
template<uint D>
auto in(index_space_t<D> sel, index_space_t<D> space)
{
    return array(index_space_contains(sel), space);
}




/**
 * Array reductions
 *
 */
template<uint D, class F, class R, class E, class A,
    typename T = typename array_t<D, F>::value_type,
    typename B = typename A::allocation_t>
T reduce(const array_t<D, F>& a, R reducer, T start, E& executor, A& allocator)
{
    if (! executor.has_async_reduction())
    {
        auto b = cache(a, executor, allocator);
        return executor.reduce(*b.buffer(), reducer, start, allocator);
    }
    else
    {
        auto num_devices = executor.num_devices();
        auto subarrays = vec_t<array_t<D, lookup_t<D, T, B>>, VAPOR_MAX_DEVICES>{};
        auto subresult = vec_t<B, VAPOR_MAX_DEVICES>{};

        for (int device = 0; device < num_devices; ++device)
        {
            auto subspace = a.space().subspace(num_devices, device);
            subarrays[device] = cache_async(a[subspace], device, executor, allocator);
        }
        for (int device = 0; device < num_devices; ++device)
        {
            const auto& b = subarrays[device];
            subresult[device] = executor.reduce_async(*b.buffer(), reducer, start, allocator);
        }
        auto result = start;

        for (int device = 0; device < num_devices; ++device)
        {
            result = reducer(result, subresult[device]->template read<T>(0));
        }
        return result;
    }
}




/**
 * Convenience methods for common reductions
 *
 */
template<uint D, class F, class E, class A, typename T = typename array_t<D, F>::value_type>
T min(const array_t<D, F>& a, E& executor, A& allocator)
{
    auto r = [] HD (const T& a, const T& b) { return a < b ? a : b; };
    return reduce(a, r, std::numeric_limits<T>::max(), executor, allocator);
}
template<uint D, class F>
auto min(const array_t<D, F>& a)
{
    return min(a, Runtime::executor(), Runtime::allocator());
}
template<uint D, class F, class E, class A, typename T = typename array_t<D, F>::value_type>
T max(const array_t<D, F>& a, E& executor, A& allocator)
{
    auto r = [] HD (const T& a, const T& b) { return a > b ? a : b; };
    return reduce(a, r, std::numeric_limits<T>::min(), executor, allocator);
}
template<uint D, class F>
auto max(const array_t<D, F>& a)
{
    return max(a, Runtime::executor(), Runtime::allocator());
}
template<uint D, class F, class E, class A, typename T = typename array_t<D, F>::value_type>
T sum(const array_t<D, F>& a, E& executor, A& allocator)
{
    auto r = [] HD (const T& a, const T& b) { return a + b; };
    return reduce(a, r, T(), executor, allocator);
}
template<uint D, class F>
auto sum(const array_t<D, F>& a)
{
    return sum(a, Runtime::executor(), Runtime::allocator());
}

} // namespace vapor
