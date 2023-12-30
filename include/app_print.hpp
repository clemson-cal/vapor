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
static inline void print(const char *s, FILE* file=stdout)
{
    fprintf(file, "%s", s);
}
static inline void print(int a, FILE* file=stdout)
{
    fprintf(file, "%d", a);
}
static inline void print(uint a, FILE* file=stdout)
{
    fprintf(file, "%u", a);
}
static inline void print(unsigned long n, FILE* file=stdout)
{
    fprintf(file, "%lu", n);
}
static inline void print(float a, FILE* file=stdout)
{
    fprintf(file, "%f", a);
}
static inline void print(double a, FILE* file=stdout)
{
    fprintf(file, "%lf", a);
}
static inline void print(bool b, FILE* file=stdout)
{
    fprintf(file, "%s", b ? "true" : "false");
}




/**
 * Print function for std::string (opt-in)
 * 
 */
#ifdef VAPOR_STD_STRING
#include <string>
static inline void print(const std::string& s, FILE* file=stdout)
{
    print(s.data(), file);
}
#endif




/**
 * Print function for std::vector<T> (opt-in)
 * 
 */
#ifdef VAPOR_STD_VECTOR
#include <vector>
template<typename T> void print(const std::vector<T>& v, FILE* file=stdout)
{
    print("std::vector(", file);
    for (size_t n = 0; n < v.size(); ++n)
    {
        print(v[n], file);
        if (n != v.size() - 1) print(" ", file);
    }
    print(")", file);
}
#endif




/**
 * Print functions for vapor types
 *
 * Note that vec_t<char, S> has a special meaning to enable use of that type
 * as a statically allocated string.
 */
template<typename T, uint S> void print(const vec_t<T, S>& v, FILE* file=stdout)
{
    for (size_t n = 0; n < S; ++n)
    {
        print(v[n], file);
        if (n != S - 1) print(" ", file);
    }
}
template<uint S> void print(vec_t<char, S> v, FILE* file=stdout)
{
    fprintf(file, "%s", v.data);
}
template<uint D> void print(const index_space_t<D>& space, FILE* file=stdout)
{
    print("start: ", file);
    print(space.i0, file);
    print(" ", file);
    print("shape: ", file);
    print(space.di, file);
}
template<uint D, typename T> void print(const array_t<D, T>& v, FILE* file=stdout)
{
    print("array(", file);
    for (size_t n = 0; n < v.size(); ++n)
    {
        print(v[n], file);
        if (n != v.size() - 1) print(" ", file);
    }
    print(")", file);
}




/**
 * Print functions for visitable type
 * 
 */
template<typename T, typename = std::enable_if_t<visit_struct::traits::is_visitable<T>::value>>
void print(const T& target, FILE* file=stdout)
{
    visit_struct::for_each(target, [file] (const char *key, const auto& val)
    {
        print(key, file);
        print(" = ", file);
        print(val, file);
        print("\n", file);
    });
}

template<typename T>
void print_to_file(const T& target, const char* filename)
{
    auto outfile = fopen(filename, "w");

    if (outfile == nullptr) {
        throw std::runtime_error(format("file %s could not be opened for writing", filename));
    }
    print(target, outfile);
    fclose(outfile);
}

} // namespace vapor
