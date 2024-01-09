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
    bool operator==(const index_space_t& other) const
    {
        return i0 == other.i0 && di == other.di;
    }

    bool operator!=(const index_space_t& other) const
    {
        return i0 != other.i0 || di != other.di;
    }

    /**
     * Return the start of this index space
     * 
     */
    ivec_t<D> start() const
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
     * Return the first index that would be visited in a traversal
     *
     */
    ivec_t<D> front() const
    {
        return i0;
    }

    /**
     * Return the final index that would be visited in a traversal
     *
     */
    ivec_t<D> back() const
    {
        return i0 + cast<int>(di) - ones_vec<int, D>();
    }

    /**
     * Return true if this index space has a zero-extent on any axis
     *
     */
    bool empty() const
    {
        return size() == 0;
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
    HD bool contains(ivec_t<D> i) const
    {
        for (uint axis = 0; axis < D; ++axis)
        {
            if (i[axis] < i0[axis] || i[axis] >= i0[axis] + int(di[axis]))
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
        return other.empty() || (contains(other.front()) && contains(other.back()));
    }

    /**
     * An index sub-space representing the n-th partition of this one on an axis
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
     * An index sub-space in a D-dimensional decomposition of this one
     *
     * The shape argument is analogous to the num_partitions argument in the
     * one-dimensional version of this function above, and the coords argument
     * is analogous to the which_partition argument.
     */
    index_space_t subspace(uvec_t<D> shape, uvec_t<D> coords) const
    {
        auto space = *this;

        for (uint axis = 0; axis < D; ++axis)
        {
            space = space.subspace(shape[axis], coords[axis], axis);
        }
        return space;
    }

    index_space_t<D> shift(int amount, uint axis) const
    {
        auto result = *this;
        result.i0[axis] += amount;
        return result;
    }

    index_space_t nudge(ivec_t<D> lower, ivec_t<D> upper) const
    {
        auto j0 = i0;
        auto dj = di;

        for (uint axis = 0; axis < D; ++axis)
        {
            j0[axis] += lower[axis];
            dj[axis] += upper[axis] - lower[axis];
        }
        return {j0, dj};
    }

    /**
     * Remove the given number of indexes from the lower and upper extent
     * 
     */
    index_space_t contract(uvec_t<D> count) const
    {
        return nudge(cast<int>(count), -cast<int>(count));
    }

    /**
     * Remove indexes from all axes
     * 
     */
    index_space_t contract(uint count) const
    {
        return contract(uniform_vec<D>(count));
    }

    /**
     * Add the given number of indexes from the lower and upper extent
     * 
     */
    index_space_t expand(uvec_t<D> count) const
    {
        return nudge(-cast<int>(count), cast<int>(count));
    }

    /**
     * Add indexes to all axes
     * 
     */
    index_space_t expand(uint count) const
    {
        return expand(uniform_vec<D>(count));
    }

    index_space_t<D> with_start(ivec_t<D> new_start) const
    {
        auto result = *this;
        result.i0 = new_start;
        return result;
    }

    index_space_t<D> upper(uint amount, uint axis) const
    {
        auto result = *this;
        result.i0[axis] = i0[axis] + di[axis] - amount;
        result.di[axis] = amount;
        return result;
    }

    index_space_t<D> lower(uint amount, uint axis) const
    {
        auto result = *this;
        result.di[axis] = amount;
        return result;
    }

    ivec_t<D> i0;
    uvec_t<D> di;
};




/**
 * Construct an index space starting at the zero-vector, with the given shape
 * 
 */
template<uint D>
auto index_space(uvec_t<D> di)
{
    return index_space_t<D>{zeros_ivec<D>(), di};
}




/**
 * Construct an index space with the given start and shape
 * 
 */
template<uint D>
auto index_space(ivec_t<D> i0, uvec_t<D> di)
{
    return index_space_t<D>{i0, di};
}

} // namespace vapor
