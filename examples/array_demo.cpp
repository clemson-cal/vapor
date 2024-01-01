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




#include <cstdio>
#include <vector>
#include "vapor/array.hpp"
#include "vapor/executor.hpp"
#include "vapor/index_space.hpp"
#include "vapor/memory.hpp"
#include "vapor/print.hpp"
#include "vapor/vec.hpp"

using namespace vapor;




void type_constructors()
{
    printf("construct some basic types...\n");
    auto v = vec(0.0, 1.0, 2.0);
    printf("v = %f %f %f\n", v[0], v[1], v[2]);
}

void make_cached_1d_array()
{
    printf("create and cache a 1d array...\n");
    auto exec1 = default_executor_t();
    auto exec2 = std::move(exec1);
    auto alloc = pool_allocator_t();

    int N = 10;
    auto a = array([] HD (uvec_t<1> i) { return 2 * i[0]; }, N);
    auto b = a * 2;
    auto c = b.cache(exec2, alloc);

    for (int i = 0; i < N; ++i)
    {
        printf("[%u]: a=2*%u=%u; b=%u=%u\n", i, i, a[i], b[i], c[i]);
    }
}

void map_subset()
{
    printf("map only a subset of an array...\n");

    int N = 5;
    auto first_two = index_space(uvec(2));
    auto a = range(N);
    auto b = a.at(first_two).map([] HD (uint i) { return 0; });
    auto c = a.at(first_two) * 0; // equivalent to b above
    auto d = a.at(first_two).set(0); // also equivalent

    for (int i = 0; i < N; ++i)
    {
        printf("[%u]: %u=%u=%u=%u\n", i, b[i], i < 2 ? 0 : i, c[i], d[i]);
    }
    printf("range(5)[2:4].start[0]=%d=2\n", a[index_space(uvec(2), uvec(2))].start()[0]);
    printf("range(5)[2:4].shape[0]=%d=2\n", a[index_space(uvec(2), uvec(2))].shape()[0]);
}

void array_reductions()
{
    auto exec = cpu_executor_t();
    auto alloc = shared_ptr_allocator_t();
    auto a = range(10).cache(exec, alloc);
    printf("the maximum value of range(10) is %d\n", max(a, exec));

    auto b = array([] (auto) { return vec(1.0, 1.0); }, uvec(10)).cache(exec, alloc);
    auto s = sum(b, exec);
    printf("the sum of 10 elements of vec(1.0, 1.0) is (%.1lf %.1lf)\n", s[0], s[1]);
}

void generator_arrays()
{
    auto d = zeros<int>(uvec(4, 8, 12));
    printf("an array of zeros with shape (%d %d %d)\n",
        d.shape()[0], d.shape()[1], d.shape()[2]);

    auto e = ones<int>(uvec(4, 8, 12));
    printf("an array of ones anywhere is %d\n", e[uvec(0, 0, 0)]);

    auto i = indices(uvec(4, 8, 12));
    printf("an array of indexes, at index (3 7 11) is (%d %d %d)\n",
        i[uvec(3, 7, 11)][0], i[uvec(3, 7, 11)][1], i[uvec(3, 7, 11)][2]);
}

void construct_pointer_types()
{
    printf("use library smart pointers\n");
    auto p = managed_memory_ptr_t<int>(22);
    printf("%d=22\n", *p);
    auto q = std::move(p);
    *q = 23;
    printf("%d=23\n", *q);

    auto ints = std::vector<managed_memory_ptr_t<int>>();
    ints.push_back(std::move(q));
    *ints[0] = 24;

    printf("%d=24\n", *ints[0]);
}

void decompose_index_space()
{
    printf("decompose an index space\n");

    index_space(uvec(5, 5), uvec(10, 20)).decompose(4, [] (auto space)
    {
        print(space);
        print("\n");
    });
}

int main()
{
    type_constructors();
    make_cached_1d_array();
    map_subset();
    array_reductions();
    generator_arrays();
    construct_pointer_types();
    decompose_index_space();
    return 0;
}
