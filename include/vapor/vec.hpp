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
#include <type_traits>
#include <cmath>
#include "compat.hpp"

namespace vapor {




template<typename T, uint S>
struct vec_t
{
    HD T& operator[](uint i)
    {
        #ifdef VAPOR_VEC_BOUNDS_CHECK
        assert(i >= 0 && i < S);
        #endif
        return data[i];
    }
    HD const T& operator[](uint i) const
    {
        #ifdef VAPOR_VEC_BOUNDS_CHECK
        assert(i >= 0 && i < S);
        #endif
        return data[i];
    }
    HD operator T*()
    {
        return data;
    }
    HD operator const T*() const
    {
        return data;
    }
    T data[S] = {{}};
};

template<uint S> using uvec_t = vec_t<uint, S>;
template<uint S> using ivec_t = vec_t<int, S>;
template<uint S> using dvec_t = vec_t<double, S>;




/**
 * Iterators
 *
 *
 */
template<typename T, uint S>
auto begin(const vec_t<T, S>& v)
{
    return v.data;
}

template<typename T, uint S>
auto end(const vec_t<T, S>& v)
{
    return v.data + S;
}




/**
 * Constructors
 *
 *
 */
template<typename... Args>
HD auto vec(Args... args)
{
    return vec_t<std::common_type_t<Args...>, sizeof...(Args)>{args...};
}
template<typename... Args>
HD auto ivec(Args... args)
{
    return ivec_t<sizeof...(Args)>{int(args)...};
}
template<typename... Args>
HD auto uvec(Args... args)
{
    return uvec_t<sizeof...(Args)>{uint(args)...};
}
template<typename... Args>
HD auto dvec(Args... args)
{
    return dvec_t<sizeof...(Args)>{double(args)...};
}
template<typename T, uint S>
HD auto zeros_vec()
{
    return vec_t<T, S>{0};
}
template<uint S>
HD auto zeros_uvec()
{
    return zeros_vec<uint, S>();
}
template<uint S>
HD auto zeros_ivec()
{
    return zeros_vec<int, S>();
}
template<uint S>
HD auto zeros_dvec()
{
    return zeros_vec<double, S>();
}
template<typename T, uint S>
HD auto ones_vec()
{
    auto res = vec_t<T, S>{};
    for (uint i = 0; i < S; ++i)
    {
        res[i] = 1;
    }
    return res;
}
template<uint S>
HD auto ones_uvec()
{
    return ones_vec<uint, S>();
}
template<uint S>
HD auto ones_ivec()
{
    return ones_vec<int, S>();
}
template<uint S>
HD auto ones_dvec()
{
    return ones_vec<double, S>();
}
template<uint S>
HD auto range_uvec()
{
    auto res = uvec_t<S>{};

    for (uint n = 0; n < S; ++n)
    {
        res[n] = n;
    }
    return res;
}
template<uint S, typename T>
HD auto uniform_vec(T val)
{
    auto res = vec_t<T, S>{};

    for (uint n = 0; n < S; ++n)
    {
        res[n] = val;
    }
    return res;
}




/**
 * Arithmetic operators, between vec and vec, and between vec and scalar
 * 
 */
template<typename T, uint S> HD auto operator+(const vec_t<T, S>& x) { vec_t<T, S> r; for (uint m = 0; m < S; ++m) r[m] = +x[m]; return r; }
template<typename T, uint S> HD auto operator-(const vec_t<T, S>& x) { vec_t<T, S> r; for (uint m = 0; m < S; ++m) r[m] = -x[m]; return r; }
template<typename T, typename U, uint S> HD auto operator+(const vec_t<T, S>& x, const vec_t<U, S>& y) { vec_t<decltype(T() + U()), S> r; for (uint m = 0; m < S; ++m) r[m] = x[m] + y[m]; return r; }
template<typename T, typename U, uint S> HD auto operator-(const vec_t<T, S>& x, const vec_t<U, S>& y) { vec_t<decltype(T() - U()), S> r; for (uint m = 0; m < S; ++m) r[m] = x[m] - y[m]; return r; }
template<typename T, typename U, uint S> HD auto operator*(const vec_t<T, S>& x, const vec_t<U, S>& y) { vec_t<decltype(T() * U()), S> r; for (uint m = 0; m < S; ++m) r[m] = x[m] * y[m]; return r; }
template<typename T, typename U, uint S> HD auto operator/(const vec_t<T, S>& x, const vec_t<U, S>& y) { vec_t<decltype(T() / U()), S> r; for (uint m = 0; m < S; ++m) r[m] = x[m] / y[m]; return r; }
template<typename T, typename U, uint S> HD auto operator+(const vec_t<T, S>& x, U b) { vec_t<decltype(T() + U()), S> r; for (uint m = 0; m < S; ++m) r[m] = x[m] + b; return r; }
template<typename T, typename U, uint S> HD auto operator-(const vec_t<T, S>& x, U b) { vec_t<decltype(T() - U()), S> r; for (uint m = 0; m < S; ++m) r[m] = x[m] - b; return r; }
template<typename T, typename U, uint S> HD auto operator*(const vec_t<T, S>& x, U b) { vec_t<decltype(T() * U()), S> r; for (uint m = 0; m < S; ++m) r[m] = x[m] * b; return r; }
template<typename T, typename U, uint S> HD auto operator/(const vec_t<T, S>& x, U b) { vec_t<decltype(T() / U()), S> r; for (uint m = 0; m < S; ++m) r[m] = x[m] / b; return r; }
template<typename T, typename U, uint S> HD auto operator+(U b, const vec_t<T, S>& x) { vec_t<decltype(U() + T()), S> r; for (uint m = 0; m < S; ++m) r[m] = b + x[m]; return r; }
template<typename T, typename U, uint S> HD auto operator-(U b, const vec_t<T, S>& x) { vec_t<decltype(U() - T()), S> r; for (uint m = 0; m < S; ++m) r[m] = b - x[m]; return r; }
template<typename T, typename U, uint S> HD auto operator*(U b, const vec_t<T, S>& x) { vec_t<decltype(U() * T()), S> r; for (uint m = 0; m < S; ++m) r[m] = b * x[m]; return r; }
template<typename T, typename U, uint S> HD auto operator/(U b, const vec_t<T, S>& x) { vec_t<decltype(U() / T()), S> r; for (uint m = 0; m < S; ++m) r[m] = b / x[m]; return r; }
template<typename T, typename U, uint S> HD auto& operator+=(vec_t<T, S>& x, const vec_t<U, S>& y) { for (uint m = 0; m < S; ++m) { x[m] += y[m]; } return x; }
template<typename T, typename U, uint S> HD auto& operator-=(vec_t<T, S>& x, const vec_t<U, S>& y) { for (uint m = 0; m < S; ++m) { x[m] -= y[m]; } return x; }
template<typename T, typename U, uint S> HD auto& operator*=(vec_t<T, S>& x, const vec_t<U, S>& y) { for (uint m = 0; m < S; ++m) { x[m] *= y[m]; } return x; }
template<typename T, typename U, uint S> HD auto& operator/=(vec_t<T, S>& x, const vec_t<U, S>& y) { for (uint m = 0; m < S; ++m) { x[m] /= y[m]; } return x; }
template<typename T, typename U, uint S> HD auto& operator+=(vec_t<T, S>& x, U b) { for (uint m = 0; m < S; ++m) { x[m] += b; } return x; }
template<typename T, typename U, uint S> HD auto& operator-=(vec_t<T, S>& x, U b) { for (uint m = 0; m < S; ++m) { x[m] -= b; } return x; }
template<typename T, typename U, uint S> HD auto& operator*=(vec_t<T, S>& x, U b) { for (uint m = 0; m < S; ++m) { x[m] *= b; } return x; }
template<typename T, typename U, uint S> HD auto& operator/=(vec_t<T, S>& x, U b) { for (uint m = 0; m < S; ++m) { x[m] /= b; } return x; }

template<typename T, typename U, uint S>
HD auto operator==(const vec_t<T, S>& x, const vec_t<U, S>& y)
{
    for (uint m = 0; m < S; ++m)
    {
        if (x[m] != y[m])
        {
            return false;
        }
    }
    return true;
}

template<typename T, typename U, uint S>
HD auto operator!=(const vec_t<T, S>& x, const vec_t<U, S>& y)
{
    for (uint m = 0; m < S; ++m)
    {
        if (x[m] != y[m])
        {
            return true;
        }
    }
    return false;
}




/**
 * Return the dot product of two vec's
 *
 */
template<typename A, typename B, uint S, typename C = decltype(A() * B())>
HD auto dot(vec_t<A, S> a, vec_t<B, S> b) -> C
{
    C n = {};
    for (uint i = 0; i < S; ++i)
    {
        n += a[i] * b[i];
    }
    return n;
}

template<typename T, uint S>
HD auto sum(vec_t<T, S> a) -> T
{
    T n = {};
    for (uint i = 0; i < S; ++i)
    {
        n += a[i];
    }
    return n;
}

template<typename T, uint S>
HD auto pow(vec_t<T, S> a, T x) -> vec_t<T, S>
{
    vec_t<T, S> n;
    for (uint i = 0; i < S; ++i)
    {
        n[i] = std::pow(a[i], x);
    }
    return n;
}

/**
 * Return the cross product between two 2d vec's
 *
 */
template<typename T>
HD T cross(vec_t<T, 2> t, vec_t<T, 2> u)
{
    return t[0] * u[1] - t[1] * u[0];
}

/**
 * Return the cross product between two 3d vec's
 *
 */
template<typename T>
HD vec_t<T, 3> cross(vec_t<T, 3> t, vec_t<T, 3> u)
{
    return make_vec(
        t[0] * u[1] * t[1] * u[0],
        t[1] * u[2] * t[2] * u[1],
        t[2] * u[0] * t[0] * u[2]
    );
}

/**
 * Return a vec, with the value appended (size is 1 greater)
 *
 */
template<typename T, uint S>
HD auto append(vec_t<T, S> t, T val) -> vec_t<T, S + 1>
{
    auto res = vec_t<T, S + 1>{};

    for (uint n = 0; n < S; ++n)
    {
        res[n] = t[n];
    }
    res[S] = val;
    return res;
}

/**
 * Return a vec, with the value prepended (S is 1 greater)
 *
 *
 */
template<typename T, uint S>
HD vec_t<T, S + 1> prepend(vec_t<T, S> t, T val)
{
    auto res = vec_t<T, S + 1>{};

    for (uint n = 0; n < S; ++n)
    {
        res[n + 1] = t[n];
    }
    res[0] = val;
    return res;
}

/**
 * Return the dot product of two uivec's
 *
 * The inputs do not need to have the same length; missing values are treated
 * as zeros.
 *
 *
 */
template<uint S>
HD uint product(uvec_t<S> t)
{
    uint n = 1;

    for (uint i = 0; i < S; ++i)
    {
        n *= t[i];
    }
    return n;
}

/**
 * Return a uivec that is offset along one index by a certain amount
 *
 */
template<uint S>
HD uvec_t<S> delta(uvec_t<S> t, uint index, int change)
{
    t[index] += change;
    return t;
}

/**
 * Type cast to a different type of vec
 *
 */
template<typename U, typename T, uint S>
HD vec_t<U, S> cast(const vec_t<T, S>& a)
{
    auto result = vec_t<U, S>();

    for (uint i = 0; i < S; ++i)
    {
        result[i] = a[i];
    }
    return result;
}

/**
 * Return true if the vec is a permutation of S integers
 *
 */
template<uint S>
auto is_permutation(const vec_t<uint, S>& a)
{
    for (uint n = 0; n < S; ++n)
    {
        int count = 0;

        for (uint m = 0; m < S; ++m)
        {
            if (a[m] == n)
            {
                count += 1;
            }
        }
        if (count != 1)
        {
            return false;
        }
    }
    return true;
}

/**
 * Return a vec, permuted by the given permutation vec
 *
 */
template<typename T, uint S>
auto permute(const vec_t<T, S>& v, vec_t<uint, S> permutation)
{
    auto w = vec_t<T, S>{};

    for (uint m = 0; m < S; ++m)
    {
        w[m] = v[permutation[m]];
    }
    return w;
}

/**
 * Return a vec, inversely permuted by the given permutation vec
 *
 */
template<typename T, uint S>
auto reverse_permute(const vec_t<T, S>& v, vec_t<uint, S> permutation)
{
    auto w = vec_t<T, S>{};

    for (uint m = 0; m < S; ++m)
    {
        w[permutation[m]] = v[m];
    }
    return w;
}

/**
 * Generate standard C-style array strides for a given array shape.
 *
 */
template<uint S>
HD uvec_t<S> strides_row_major(uvec_t<S> shape)
{
    auto result = uvec_t<S>{};

    if constexpr (S > 0)
    {
        result[S - 1] = 1;
    }

    if constexpr (S > 1)
    {
        for (int n = S - 2; n >= 0; --n)
        {
            result[n] = result[n + 1] * shape[n + 1];
        }
    }
    return result;
}

} // namespace vapor
