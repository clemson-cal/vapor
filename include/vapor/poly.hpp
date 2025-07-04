/**
================================================================================
Copyright 2023 - 2024, Jonathan Zrake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

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
#include "vec.hpp"

namespace vapor {
namespace poly {




template<typename T, typename U, uint S, typename V = decltype(T() * U())>
auto eval(const vec_t<T, S>& p, U x) -> V {
    auto r = V(0);
    for (int i = S - 1; i >= 0; --i) {
        r *= x;
        r += p[i];
    }
    return r;
}

template<typename T, uint S>
auto integ(const vec_t<T, S>& p) -> vec_t<T, S + 1> {
    auto r = vec_t<T, S + 1>{};
    for (uint i = 0; i < S; ++i) {
        r[i + 1] = p[i] / T(i + 1);
    }
    return r;
}

template<typename T, uint S>
auto integ(const vec_t<T, S>& p, T a, T b) -> T {
    auto r = integ(p);
    return eval(r, b) - eval(r, a);
}

template<typename T, uint S, typename = std::enable_if_t<(S > 1)>>
auto diff(const vec_t<T, S>& p) -> vec_t<T, S - 1> {
    auto y = vec_t<T, S - 1>{};
    for (uint i = 1; i < S; ++i) {
        y[i - 1] = p[i] * T(i);
    }
    return y;
}

template<typename T, uint S1, uint S2>
auto mul(const vec_t<T, S1>& a, const vec_t<T, S2>& b) -> vec_t<T, S1 + S2 - 1> {
    auto r = vec_t<T, S1 + S2 - 1>{};
    for (uint i = 0; i < S1; ++i) {
        for (uint j = 0; j < S2; ++j) {
            r[i + j] += a[i] * b[j];
        }
    }
    return r;
}

template<uint M, typename T, uint N, typename = std::enable_if_t<(M >= N)>>
auto promote(const vec_t<T, N>& p) -> vec_t<T, M> {
    auto r = vec_t<T, M>{};
    for (uint i = 0; i < N; ++i) {
        r[i] = p[i];
    }
    return r;
}

template<uint N, typename T = double>
auto monomial() -> vec_t<T, N + 1> {
    auto r = vec_t<T, N + 1>{};
    r[N] = T(1);
    return r;
}

template<uint N, typename T = double>
auto legendre() -> vec_t<T, N + 1> {
    if constexpr (N == 0) {
        return monomial<0>();
    } else if constexpr (N == 1) {
        return monomial<1>();
    } else {
        // ( (2n-1) * x * P[n-1] - (n-1) * P[n-2] ) / n
        uint n = N - 1;
        auto Pn1 = legendre<N - 1, T>();
        auto Pn2 = legendre<N - 2, T>();
        auto x = monomial<1>();
        return ((T(2 * n - 1) * mul(x, Pn1)) - (T(n - 1) * promote<N + 1>(Pn2))) / T(n);
    }
}

template<typename T, uint S>
void print(const vec_t<T, S>& p) {
    printf("poly: ");
    for (uint i = 0; i < S; ++i) {
        if (i > 0) printf(" + ");
        printf("%g", double(p[i]));
        if (i > 0) printf("*x^%u", i);
    }
    printf("\n");
}

} // namespace poly
} // namespace vapor
