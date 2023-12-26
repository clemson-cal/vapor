#pragma once
#include <cstdio>
#include "core_array.hpp"
#include "core_vec.hpp"
#include "visit_struct.hpp"

namespace vapor {




/**
 * Convenience for using snprintf to write into a stack-alloc'd vec of char.
 */
template<unsigned int S=1024, typename... Args>
auto message(const char* format, Args... args)
{
    auto message = vapor::zeros_vec<char, S>();
    snprintf(message, S, format, args...);
    return message;
}




/**
 * Print functions for native types
 * 
 */
static inline void print(const char *s)
{
    printf("%s", s);
}
static inline void print(int a)
{
    printf("%d", a);
}
static inline void print(uint a)
{
    printf("%u", a);
}
static inline void print(unsigned long n)
{
    printf("%lu", n);
}
static inline void print(float a)
{
    printf("%f", a);
}
static inline void print(double a)
{
    printf("%lf", a);
}
static inline void print(bool b)
{
    printf("%s", b ? "true" : "false");
}




/**
 * Print function for std::string (opt-in)
 * 
 */
#ifdef VAPOR_STD_STRING
#include <string>
static inline void print(const std::string& s)
{
    print(s.data());
}
#endif




/**
 * Print function for std::vector<T> (opt-in)
 * 
 */
#ifdef VAPOR_STD_VECTOR
#include <vector>
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
#endif




/**
 * Print functions for vapor types
 * 
 */
template<typename T, uint S> void print(vapor::vec_t<T, S> v)
{
    for (size_t n = 0; n < S; ++n)
    {
        print(v[n]);
        if (n != S - 1) print(" ");
    }
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




/**
 * Print functions for visitable type
 * 
 */
template<typename T, typename = std::enable_if_t<visit_struct::traits::is_visitable<T>::value>>
void print(const T& target)
{
    visit_struct::for_each(target, [] (const char *key, const auto& val)
    {
        vapor::print(key);
        vapor::print(": ");
        vapor::print(val);
        vapor::print("\n");
    });
}

} // namespace vapor
