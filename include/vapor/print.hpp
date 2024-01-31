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
#ifdef VAPOR_STD_VECTOR
#include <vector>
#endif
#ifdef VAPOR_STD_STRING
#include <string>
#endif
#ifdef VAPOR_STD_MAP
#include <map>
#endif
#include <cstdio>
#include "array.hpp"
#include "vec.hpp"
#include "visit_struct/visit_struct.hpp"

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
    fprintf(file, "%g", a);
}
static inline void print(double a, FILE* file=stdout)
{
    fprintf(file, "%g", a);
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
template<typename T> void print(const std::vector<T>& v, FILE* file=stdout)
{
    for (size_t n = 0; n < v.size(); ++n)
    {
        print(v[n], file);
        if (n != v.size() - 1) print(" ", file);
    }
}
#endif




/**
 * Print function for std::map<T, U> (opt-in)
 * 
 */
#ifdef VAPOR_STD_MAP
template<typename T, typename U> void print(const std::map<T, U>& d, FILE* file=stdout)
{
    auto n = size_t(0);
    print("{", file);
    for (const auto& [key, val] : d)
    {
        print(key, file);
        print(":", file);
        print(val, file);
        if (n++ != d.size() - 1) print(" ", file);
    }
    print("}", file);
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
    for (size_t i = 0; i < S; ++i)
    {
        if (v[i] == '\0') {
            break;
        }
        fprintf(file, "%c", v[i]);
    }
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

template<typename... Args>
void print(Args... args)
{
    (print(args),...);
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
