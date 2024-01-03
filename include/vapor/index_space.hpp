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
#include "vec.hpp"

namespace vapor {




/**
 * Return an integer interval, representing the n-th partition of a larger range
 *
 * The interval is represented as (start, size). If the number of partitions
 * does not divide the number of elements, then the remainder is distributed
 * to the leading intervals.
 * 
 * Examples:
 * 
 * partition_interval(10, 101, 0) = (0, 11)
 * partition_interval(10, 101, 1) = (11, 10)
 * 
 */
inline static uvec_t<2> partition_interval(uint num_partitions, uint num_elements, uint which_partition)
{
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
    auto large_partition_size = num_elements / num_partitions + 1;
    auto small_partition_size = num_elements / num_partitions;
    auto num_large_partitions = num_elements % num_partitions;
    auto n_large = min2(which_partition, num_large_partitions);
    auto n_small = max2(which_partition - n_large, 0);
    auto i0 = n_large * large_partition_size + n_small * small_partition_size;
    auto di = which_partition < num_large_partitions ? large_partition_size : small_partition_size;
    return vec(i0, di);
#undef min2
#undef max2
}




/**
 * An n-dimensional index space
 *
 *
 * 
 */
template<uint D>
struct index_space_t
{
    /**
     * Remove the given number of indexes from the lower and upper extent
     * 
     */
    index_space_t contract(uint count) const
    {
        auto j0 = i0;
        auto dj = di;

        for (uint axis = 0; axis < D; ++axis)
        {
            j0[axis] += count;
            dj[axis] -= count * 2;
        }
        return {j0, dj};
    }

    /**
     * Add the given number of indexes to the lower and upper extent
     * 
     */
    index_space_t expand(uint count) const
    {
        auto j0 = i0;
        auto dj = di;

        for (uint axis = 0; axis < D; ++axis)
        {
            j0[axis] -= count;
            dj[axis] += count * 2;
        }
        return {j0, dj};
    }

    /**
     * Extend the upper extent so the space is like a space of vertices
     * 
     */
    index_space_t vertices() const
    {
        return {i0, di + ones_uvec<D>()};
    }

    uvec_t<D> shape() const
    {
        return di;
    }

    HD bool contains(uvec_t<D> i) const
    {
        for (uint axis = 0; axis < D; ++axis)
        {
            if (i[axis] < i0[axis] || i[axis] >= i0[axis] + di[axis])
            {
                return false;
            }
        }
        return true;
    }

    /**
     * An index space representing the n-th partition of this one on an axis
     * 
     */
    index_space_t subspace(uint num_partitions, uint which_partition, uint axis=0) const
    {
        auto partition = partition_interval(num_partitions, di[axis], which_partition);
        auto space = *this;
        space.i0[axis] = partition[0] + i0[axis];
        space.di[axis] = partition[1];
        return space;
    }

    uvec_t<D> i0;
    uvec_t<D> di;
};




template<uint D>
auto index_space(uvec_t<D> di)
{
    return index_space_t<D>{zeros_uvec<D>(), di};
}

template<uint D>
auto index_space(uvec_t<D> i0, uvec_t<D> di)
{
    return index_space_t<D>{i0, di};
}

} // namespace vapor
