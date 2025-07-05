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
template<typename T, uint M>
HD auto identity_mat()
{
    matrix_t<T, M, M> result;
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < M; ++j)
            result(i, j) = (i == j) ? T(1) : T(0);
    return result;
}

template<typename T, typename U, uint M, uint N>
HD auto operator*(const matrix_t<T, M, N>& a, U x)
{
    matrix_t<decltype(T() * U()), M, N> r;
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < N; ++j)
            r(i, j) = a(i, j) * x;
    return r;
}

template<typename T, typename U, uint M, uint N>
HD auto operator/(const matrix_t<T, M, N>& a, U x)
{
    matrix_t<decltype(T() * U()), M, N> r;
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < N; ++j)
            r(i, j) = a(i, j) / x;
    return r;
}

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
    vec_t<decltype(T() * U()), M> c;

    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < N; ++j)
            c[i] += a(i, j) * b[j];
    return c;
}

template<typename T, typename U, uint M, uint N>
HD auto matmul(const vec_t<T, M> &a, const matrix_t<U, M, N> &b)
{
    vec_t<decltype(T() * U()), N> c = {};

    for (uint j = 0; j < N; ++j)
        for (uint i = 0; i < M; ++i)
            c[j] += a[i] * b(i, j);
    return c;
}




template<typename T, uint N>
HD auto inverse(const matrix_t<T, N, N>& A) -> matrix_t<T, N, N> {
    matrix_t<T, N, N> a = A;
    matrix_t<T, N, N> inv = identity_mat<T, N>();

    for (uint i = 0; i < N; ++i) {
        // Find pivot
        uint pivot = i;
        T max_val = std::abs(a(i, i));
        for (uint j = i + 1; j < N; ++j) {
            if (std::abs(a(j, i)) > max_val) {
                max_val = std::abs(a(j, i));
                pivot = j;
            }
        }
        if (max_val == T(0)) {
            #ifdef __CUDA_ARCH__
            return identity_mat<T, N>();
            #else
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
            #endif
        }
        // Swap rows if needed
        if (pivot != i) {
            for (uint k = 0; k < N; ++k) {
                std::swap(a(i, k), a(pivot, k));
                std::swap(inv(i, k), inv(pivot, k));
            }
        }
        // Normalize pivot row
        T diag = a(i, i);
        for (uint k = 0; k < N; ++k) {
            a(i, k) /= diag;
            inv(i, k) /= diag;
        }
        // Eliminate other rows
        for (uint j = 0; j < N; ++j) {
            if (j == i) continue;
            T factor = a(j, i);
            for (uint k = 0; k < N; ++k) {
                a(j, k) -= factor * a(i, k);
                inv(j, k) -= factor * inv(i, k);
            }
        }
    }
    return inv;
}

} // namespace vapor
