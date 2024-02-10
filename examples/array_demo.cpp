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
#include "vapor/mat.hpp"
#include "vapor/memory.hpp"
#include "vapor/print.hpp"
#include "vapor/vec.hpp"
#include "vapor/optional.hpp"

using namespace vapor;




void type_constructors()
{
    printf("construct simple vec's...\n");
    auto u = vec(0.0, 1.0, 2.0);
    auto v = u * 2.0;
    printf("demo higher-order-dot products...\n");
    auto id = vec(vec(1.0, 0.0, 0.0), vec(0.0, 1.0, 0.0), vec(0.0, 0.0, 1.0));
    auto w = dot(id, v);
    printf("w = 2 * u = %f %f %f\n", w[0], w[1], w[2]);
    printf("v = id.w  = %f %f %f\n", v[0], v[1], v[2]);

    printf("test matrix inversion and multiplication...\n");
    auto a = matrix_t<double, 3, 3>{{
        {+1.0, +0.1, +0.0},
        {-0.1, +1.0, +0.1},
        {+0.2, +0.0, +2.0},
    }};
    auto b = inverse(a);
    auto c = matmul(a, b);
    printf("a      = %+.2f %+.2f %+.2f\n", a(0, 0), a(0, 1), a(0, 2));
    printf("         %+.2f %+.2f %+.2f\n", a(1, 0), a(1, 1), a(1, 2));
    printf("         %+.2f %+.2f %+.2f\n", a(2, 0), a(2, 1), a(2, 2));
    printf("inv(a) = %+.2f %+.2f %+.2f\n", b(0, 0), b(0, 1), b(0, 2));
    printf("         %+.2f %+.2f %+.2f\n", b(1, 0), b(1, 1), b(1, 2));
    printf("         %+.2f %+.2f %+.2f\n", b(2, 0), b(2, 1), b(2, 2));
    printf("I =      %+.6e %+.6e %+.6e\n", c(0, 0), c(0, 1), c(0, 2));
    printf("         %+.6e %+.6e %+.6e\n", c(1, 0), c(1, 1), c(1, 2));
    printf("         %+.6e %+.6e %+.6e\n", c(2, 0), c(2, 1), c(2, 2));
}

void use_optional()
{
    auto a = some(10);
    auto b = a.map([] HD (int a) { return a + 1; });
    printf("this optional should have a value: %d\n", b.has_value());
    printf("this optional should not: %d\n", none<int>().has_value());

    try {
        auto c = range(100).map([] HD (int i)
        {
            return i >= 90 ? none<int>() : some(i);
        }).cache_unwrap();
    }
    catch (const cache_unwrap_exception& e) {
        printf("%s (%d) -- [expect 10 failures]\n", e.what(), e.num_failures());
    }
}

void make_cached_1d_array()
{
    printf("create and cache a 1d array...\n");
    auto exec1 = default_executor_t();
    auto exec2 = std::move(exec1);
    auto alloc = pool_allocator_t();

    int N = 10;
    auto a = array([] HD (ivec_t<1> i) { return 2 * i[0]; }, N);
    auto b = a * 2;
    auto c = cache(b, exec2, alloc);

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
    auto b = a.at(first_two).map([] HD (int i) { return 0; });
    auto c = a.at(first_two) * 0; // equivalent to b above
    auto d = a.at(first_two).set(0); // also equivalent

    for (int i = 0; i < N; ++i)
    {
        printf("[%u]: %u=%u=%u=%u\n", i, b[i], i < 2 ? 0 : i, c[i], d[i]);
    }
    printf("range(5)[2:4].start[0]=%d=2\n", a[index_space(ivec(2), uvec(2))].start()[0]);
    printf("range(5)[2:4].shape[0]=%d=2\n", a[index_space(ivec(2), uvec(2))].shape()[0]);
}

void array_reductions()
{
    #ifndef __CUDACC__
    auto exec = cpu_executor_t();
    auto alloc = shared_ptr_allocator_t();
    #else
    auto exec = default_executor_t();
    auto alloc = pool_allocator_t();
    #endif
    
    auto N = int(1000);
    auto a = cache(range(N), exec, alloc);
    printf("the maximum value of range(%d) is %d\n", N, max(a, exec, alloc));
    printf("the sum of range(%d) is %d\n", N, sum(a, exec, alloc));
    auto b = cache(array([] HD (ivec_t<1>) { return vec(1.0, 1.0); }, uvec(N)), exec, alloc);
    auto s = sum(b, exec, alloc);
    printf("the sum of %d elements of vec(1.0, 1.0) is (%.1lf %.1lf)\n", N, s[0], s[1]);

    printf("any of range(20) is >= 10? %d\n", any(range(20) >= 10));
    printf("all of range(20) is >= 10? %d\n", all(range(20) >= 10));
    printf("any of range(20) is == 10? %d\n", any(range(20) == 10));
    printf("any of range(20) is == 20? %d\n", any(range(20) == 20));
}

void generator_arrays()
{
    auto d = zeros<int>(uvec(4, 8, 12));
    printf("an array of zeros with shape (%d %d %d)\n",
        d.shape()[0], d.shape()[1], d.shape()[2]);

    auto e = ones<int>(uvec(4, 8, 12));
    printf("an array of ones anywhere is %d\n", e[ivec(0, 0, 0)]);

    auto i = indices(uvec(4, 8, 12));
    printf("an array of indexes, at index (3 7 11) is (%d %d %d)\n",
        i[ivec(3, 7, 11)][0], i[ivec(3, 7, 11)][1], i[ivec(3, 7, 11)][2]);
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

    auto space = index_space(ivec(5, 5), uvec(10, 20));

    for (int n = 0; n < 4; ++n)
    {
        print(space.subspace(4, n));
        print("\n");
    }
    print("the index space front is ", space.front(), "\n");
    print("the index space back is ", space.back(), "\n");
    print("an index space contains itself? ", space.contains(space), "\n");
    print("an index space contains the empty space? ", space.contains(index_space(zeros_uvec<2>())), "\n");
}

void readme_example_code()
{
    auto N = uint(100);
    auto s = index_space(vec(N, N));
    auto i = indices(s);
    auto h = 1.0 / N; /* grid spacing */
    auto x = i.map([h] HD (vec_t<int, 2> ij) { return ij * h - 0.5; });
    auto u = x.map([] HD (vec_t<double, 2> x) { return exp(-dot(x, x)); });
    auto del_squared_u = i[s.contract(1)].map([u, h] HD (vec_t<int, 2> ij) {
        auto i = ij[0];
        auto j = ij[1];
        return (u[vec(i + 1, j)] +
                u[vec(i, j + 1)] +
                u[vec(i - 1, j)] +
                u[vec(i, j - 1)] - 4 * u[ij]) / (h * h); /* Laplacian, Del^2(u) */
    });
    auto dt = h * 0.1;
    auto du = del_squared_u * dt;
    auto u_next = (u.at(s.contract(1)) + du).cache();
}

int main()
{
    type_constructors();
    use_optional();
    make_cached_1d_array();
    map_subset();
    array_reductions();
    generator_arrays();
    construct_pointer_types();
    decompose_index_space();
    readme_example_code();
    return 0;
}
