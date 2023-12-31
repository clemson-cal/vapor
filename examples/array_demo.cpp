#include <cstdio>
#include <vector>
#include "core_vec.hpp"
#include "core_memory.hpp"
#include "core_index_space.hpp"
#include "core_array.hpp"
#include "app_print.hpp"

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
    generator_arrays();
    construct_pointer_types();
    decompose_index_space();
    return 0;
}
