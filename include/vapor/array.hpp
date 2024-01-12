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
using memory_backed_array_t = array_t<D, lookup_t<D, T, P<managed_memory_t>>>;




/**
 * Execute an array using the given executor and allocator
 *
 */
template<uint D, class F, class E, class A,
    class T = typename array_t<D, F>::value_type,
    class R = typename A::allocation_t>
array_t<D, lookup_t<D, T, R>> cache(const array_t<D, F>& a, E& executor, A& allocator)
{
    auto memory = allocator.allocate(a.size() * sizeof(T));
    auto data = (T*) memory->data();
    auto start = a.start();
    auto stride = strides_row_major(a.shape());
    auto table = lookup(start, stride, data, memory);

    executor.loop(a.space(), [start, stride, data, a] HD (ivec_t<D> i)
    {
        data[dot(stride, i - start)] = a[i];
    });
    return array(table, a.space(), data);
}




/**
 * Convenience cache function using the global executor and allocator
 * 
 */
template<uint D, class F,
    class T = typename array_t<D, F>::value_type,
    class R = typename default_allocator_t::allocation_t>
array_t<D, lookup_t<D, T, R>> cache(const array_t<D, F>& a)
{
    return cache(a, Runtime::executor(), Runtime::allocator());
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
        return _data;
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
    template<class E, class A>
	auto cache(E& executor, A& allocator) const
	{
        return vapor::cache(*this, executor, allocator);
    }
	auto cache() const
    {
        return vapor::cache(*this);
	}
    template<bool C, class E, class A>
    auto cache_if(E& executor, A& allocator) const
    {
        if constexpr (C)
            return vapor::cache(*this, executor, allocator);
        else 
            return *this;
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
    value_type* _data = nullptr;
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
array_t<D, F> array(F f, index_space_t<D> space, T* data=nullptr)
{
    return {f, space.di, space.i0, data};
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
 * Compute the minimum value of all array elements
 *
 * Reductions require arrays to be memory-backed and contiguous
 */
template<uint D, typename T, template<typename> typename P, class E, class A>
T min(const memory_backed_array_t<D, T, P>& a, E& executor, A& allocator)
{
    return executor.reduce(
        a.data(),
        a.size(),
        [] HD (const T& a, const T& b) { return a < b ? a : b; },
        std::numeric_limits<T>::max(),
        allocator
    );
}

template<uint D, typename T, template<typename> typename P>
T min(const memory_backed_array_t<D, T, P>& a)
{
    return min(a, Runtime::executor(), Runtime::allocator());
}

/**
 * Compute the maximum value of all array elements
 *
 * Reductions require arrays to be memory-backed and contiguous
 */
template<uint D, typename T, template<typename> typename P, class E, class A>
T max(const memory_backed_array_t<D, T, P>& a, E& executor, A& allocator)
{
    return executor.reduce(
        a.data(),
        a.size(),
        [] HD (const T& a, const T& b) { return a > b ? a : b; },
        std::numeric_limits<T>::min(),
        allocator
    );
}

template<uint D, typename T, template<typename> typename P>
T max(const memory_backed_array_t<D, T, P>& a)
{
    return max(a, Runtime::executor(), Runtime::allocator());
}

/**
 * Compute the sum over all array elements
 *
 * Reductions require arrays to be memory-backed and contiguous
 */
template<uint D, typename T, template<typename> typename P, class E, class A>
T sum(const memory_backed_array_t<D, T, P>& a, E& executor, A& allocator)
{
    return executor.reduce(
        a.data(),
        a.size(),
        [] HD (const T& a, const T& b) { return a + b; },
        T(),
        allocator
    );
}

template<uint D, typename T, template<typename> typename P>
T sum(const memory_backed_array_t<D, T, P>& a)
{
    return sum(a, Runtime::executor(), Runtime::allocator());
}

} // namespace vapor
