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
#include "compat.hpp"
#include "index_space.hpp"
#include "memory.hpp"
#include "vec.hpp"

namespace vapor {




/**
 * 
 */
template<typename T>
struct constant_t
{
    template<typename U>
    HD auto operator()(U) const { return val; }
    T val;
};
template<typename T>
auto constant(T val)
{
    return constant_t<T>{val};
}




/**
 * 
 */
struct identity_t
{
    template<typename U>
    HD auto operator()(U u) const { return u; }
};
auto identity()
{
    return identity_t{};
}




/**
 * 
 */
template<typename T>
struct take_nth_t
{
    HD auto operator()(T a) const { return a[n]; }
    uint n;
};




/**
 * 
 */
template<uint D, typename T, class R>
struct lookup_t
{
    HD auto operator()(uvec_t<D> i) const { return data[dot(stride, i - start)]; }
    uvec_t<D> start;
    uvec_t<D> stride;
    T* data;
    R resource;
};
template<uint D, typename T, class R>
auto lookup(uvec_t<D> start, uvec_t<D> stride, T* data, R resource)
{
    return lookup_t<D, T, R>{start, stride, data, resource};
}




/**
 * 
 */
template<uint D, class F, class G>
struct compose_t
{
    HD auto operator()(uvec_t<D> i) const { return g(f(i)); }
    F f;
    G g;
};
template<uint D, typename F, typename G>
auto compose(F f, G g)
{
    return compose_t<D, F, G>{f, g};
}




/**
 * 
 */
template<uint D, class E, class F, class G>
struct cond_t
{
    HD auto operator()(uvec_t<D> i) const { return e(i) ? f(i) : g(i); }
    E e;
    F f;
    G g;
};
template<uint D, class E, class F, class G>
auto cond(E e, F f, G g)
{
    return cond_t<D, E, F, G>{e, f, g};
}




/**
 * 
 */
template<uint D>
struct index_space_contains_t
{
    HD auto operator()(uvec_t<D> i) const { return _space.contains(i); }
    index_space_t<D> _space;
};
template<uint D>
auto index_space_contains(index_space_t<D> sel)
{
    return index_space_contains_t<D>{sel};
}




/**
 * 
 */
template<uint D, class F, class G>
struct add_t
{
    HD auto operator()(uvec_t<D> i) const { return f(i) + g(i); }
    F f;
    G g;
};

template<uint D, class F, class G>
struct sub_t
{
    HD auto operator()(uvec_t<D> i) const { return f(i) - g(i); }
    F f;
    G g;
};

template<uint D, class F, class G>
struct mul_t
{
    HD auto operator()(uvec_t<D> i) const { return f(i) * g(i); }
    F f;
    G g;
};

template<uint D, class F, class G>
struct div_t
{
    HD auto operator()(uvec_t<D> i) const { return f(i) / g(i); }
    F f;
    G g;
};

template<uint D, class F, class G> auto add(F f, G g) { return add_t<D, F, G>{f, g}; }
template<uint D, class F, class G> auto sub(F f, G g) { return sub_t<D, F, G>{f, g}; }
template<uint D, class F, class G> auto mul(F f, G g) { return mul_t<D, F, G>{f, g}; }
template<uint D, class F, class G> auto div(F f, G g) { return div_t<D, F, G>{f, g}; }

} // namespace vapor
