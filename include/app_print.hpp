#pragma once
#include <cstdio>
#include "core_array.hpp"
#include "core_vec.hpp"
#include "visit_struct.hpp"

namespace vapor {




/**
 * Convenience for using snprintf to write into a stack-alloc'd vec of char.
 *
 * This function name is not as good as 'message', 
 */
template<unsigned int S=256, typename... Args>
auto format(const char* format, Args... args)
{
    auto message = zeros_vec<char, S>();
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
 * Note that vec_t<char, S> has a special meaning to enable use of that type
 * as a statically allocated string.
 */
template<typename T, uint S> void print(const vec_t<T, S>& v)
{
    for (size_t n = 0; n < S; ++n)
    {
        print(v[n]);
        if (n != S - 1) print(" ");
    }
}
template<uint S> void print(vec_t<char, S> v)
{
    printf("%s", v.data);
}
template<uint D> void print(const index_space_t<D>& space)
{
    print("start: ");
    print(space.i0);
    print(" ");
    print("shape: ");
    print(space.di);
}
template<uint D, typename T> void print(const array_t<D, T>& v)
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
        print(key);
        print(" = ");
        print(val);
        print("\n");
    });
}

} // namespace vapor
