#pragma once
#include <cstdio>
#include <string> // Should avoid implicit includes of string and
#include <vector> // vector, for compile time.
#include "core_array.hpp"
#include "core_vec.hpp"

namespace vapor {




/**
 * Print functions
 * 
 */
static inline void print(const char *s)
{
    printf("%s", s);
}
static inline void print(size_t n)
{
    printf("%lu", n);
}
static inline void print(int a)
{
    printf("%d", a);
}
static inline void print(uint a)
{
    printf("%u", a);
}
static inline void print(double a)
{
    printf("%lf", a);
}
static inline void print(bool b)
{
    printf("%s", b ? "true" : "false");
}
static inline void print(const std::string& s)
{
    print(s.data());
}
template<typename T, uint S> void print(vapor::vec_t<T, S> v)
{
    print("vec(");
    for (size_t n = 0; n < S; ++n)
    {
        print(v[n]);
        if (n != S - 1) print(" ");
    }
    print(")");
}
template<uint D, typename T> void print(const vapor::array_t<D, T>& v)
{
    print("array(");
    for (size_t n = 0; n < v.size(); ++n)
    {
        print(v[n]);
        if (n != v.size() - 1) print(" ");
    }
    print(")");
}
template<typename T> void print(const std::vector<T>& v)
{
    print("std::vector(");
    for (size_t n = 0; n < v.size(); ++n)
    {
        print(v[n]);
        if (n != v.size() - 1) print(" ");
    }
    print(")");
}

} // namespace vapor
