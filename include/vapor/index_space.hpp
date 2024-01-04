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
     * Return the shape of this index space
     * 
     */
    uvec_t<D> start() const
    {
        return i0;
    }

    /**
     * Return the shape of this index space
     * 
     */
    uvec_t<D> shape() const
    {
        return di;
    }

    /**
     * Return the number of elements in this index space
     * 
     */
    uint size() const
    {
        return product(di);
    }

    /**
     * Test whether the index space contains a given index
     * 
     */
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
     * Test whether another index space lies fully within this one
     * 
     */
    HD bool contains(index_space_t<D> other) const
    {
        return contains(other.front()) && contains(other.back());
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
     * Return the first index that would be visited in a traversal
     *
     */
    HD uvec_t<D> front() const
    {
        return i0;
    }

    /**
     * Return the final index that would be visited in a traversal
     *
     */
    HD uvec_t<D> back() const
    {
        return i0 + di - ones_vec<uint, D>();
    }

    uvec_t<D> i0;
    uvec_t<D> di;
};




/**
 * Construct an index space starting at the zero-vector, with the given shape
 * 
 */
template<uint D>
auto index_space(uvec_t<D> di)
{
    return index_space_t<D>{zeros_uvec<D>(), di};
}




/**
 * Construct an index space with the given start and shape
 * 
 */
template<uint D>
auto index_space(uvec_t<D> i0, uvec_t<D> di)
{
    return index_space_t<D>{i0, di};
}

} // namespace vapor
