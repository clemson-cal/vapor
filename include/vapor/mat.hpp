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

namespace vapor {




template<typename T, uint M, uint N>
struct matrix_t
{
    HD const T& operator()(uint i, uint j) const
    {
        #ifdef VAPOR_MAT_BOUNDS_CHECK
        assert(i >= 0 && i < M);
        assert(j >= 0 && j < N);
        #endif
        return data[i][j];
    }
    HD T& operator()(uint i, uint j)
    {
        #ifdef VAPOR_MAT_BOUNDS_CHECK
        assert(i >= 0 && i < M);
        assert(j >= 0 && j < N);
        #endif
        return data[i][j];
    }
    HD operator T*()
    {
        return data[0];
    }
    HD operator const T*() const
    {
        return data[0];
    }
    T data[M][N] = {{{}}};
};

template<typename T, typename U, uint M, uint N, uint K>
HD auto matmul(const matrix_t<T, M, K> &a, const matrix_t<U, K, N> &b)
{
    matrix_t<decltype(T() * U()), M, N> c;

    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < N; ++j)
            for (uint k = 0; k < K; ++k)
                c(i, j) += a(i, k) * b(k, j);
    return c;
}

template<typename T, typename U, uint M, uint N>
HD auto matmul(const matrix_t<T, M, N> &a, const vec_t<U, N> &b)
{
    vec_t<decltype(T() * U()), N> c;

    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < N; ++j)
            c[i] += a(i, j) * b[j];
    return c;
}




/**
 * Calculate the inverse of a square matrix
 *
 * This implementation is intended to be pedagogical, or for testing. It might
 * not be performant or accurate enough for production settings.
 *
 * Based on an algorithm described here:
 * 
 * http://en.wikipedia.org/wiki/LU_decomposition
 *
 * adapted from C++ code by Mike Dinolfo.
 */
template<typename T, uint M>
HD auto inverse(const matrix_t<T, M, M> &a)
{
    static const double eps = 1e-12;
    auto b = matrix_t<T, M, M>{};

    if constexpr (M == 1) {
        b(0, 0) = 1.0 / a(0, 0);
    }
    else if constexpr (M == 2) {
        auto d = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
        b(0, 0) =  a(1, 1) / d;
        b(1, 1) =  a(0, 0) / d;
        b(0, 1) = -a(0, 1) / d;
        b(1, 0) = -a(1, 0) / d;
        return b;
    }
    else {
        b = a; // copy the input matrix to output matrix

        for (uint i = 0; i < M; ++i)
        {
            if (b(i, i) < eps && b(i, i) > -eps)
            {
                b(i, i) = eps; // add eps value to diagonal if diagonal is zero
            }
        }
        for (uint i = 1; i < M; ++i)
        {
            b(i, 0) /= b(0, 0); // normalize row 0
        }
        for (uint i = 1; i < M; ++i)
        {
            for (uint j = i; j < M; ++j)
            {
                auto sum = 0.0;
                for (uint k = 0; k < i; ++k)
                {
                    sum += b(j, k) * b(k, i); // do a column of L
                }
                b(j, i) -= sum;
            }
            if (i == M - 1)
            {
                continue;
            }
            for (uint j = i + 1; j < M; ++j)
            {
                auto sum = 0.0;
                for (uint k = 0; k < i; ++k)
                {
                    sum += b(i, k) * b(k, j);
                }
                b(i, j) = (b(i, j) - sum) / b(i, i); // do a row of U
            }
        }
        auto d = 1.0; // compute the determinant, product of diag(U)

        for (uint i = 0; i < M; ++i)
        {
            d *= b(i, i);
        }
        for (uint i = 0; i < M; ++i)
        {
            for (uint j = i; j < M; ++j)
            {
                auto x = 1.0;
                if (i != j)
                {
                    x = 0.0;
                    for (uint k = i; k < j; ++k)
                    {
                        x -= b(j, k) * b(k, i);
                    }
                }
                b(j, i) = x / b(j, j); // invert L
            }
        }
        for (uint i = 0; i < M; ++i)
        {
            for (uint j = i; j < M; ++j)
            {
                auto sum = 0.0;
                if (i == j)
                {
                    continue;
                }
                for (uint k = i; k < j; ++k)
                {
                    sum += b(k, j) * (i == k ? 1.0 : b(i, k));
                }
                b(i, j) = -sum; // invert U
            }
        }
        for (uint i = 0; i < M; ++i)
        {
            for (uint j = 0; j < M; ++j)
            {
                auto sum = 0.0;
                for (uint k = i > j ? i : j; k < M; ++k)
                {
                    sum += (j == k ? 1.0 : b(j, k)) * b(k, i);
                }
                b(j, i) = sum; // final inversion
            }
        }
        return b;
    }
}

} // namespace vapor
