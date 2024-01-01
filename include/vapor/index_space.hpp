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
 * An n-dimensional index space
 *
 *
 * 
 */
template<uint D>
struct index_space_t
{
    uvec_t<D> i0;
    uvec_t<D> di;

    /**
     * Remove the given number of indexes from the lower and upper extent
     * 
     */
    index_space_t contract(uint num_guard) const
    {
        auto j0 = i0;
        auto dj = di;

        for (uint axis = 0; axis < D; ++axis)
        {
            j0[axis] += num_guard;
            dj[axis] -= num_guard * 2;
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

    template<class Yield>
    void decompose(uint num_parts, Yield yield) const
    {
        /**
         * Equitably divide the given number of elements into `num_parts` partitions.
         *
         * The sum of the partitions is `elements`. The number of partitions must be
         * less than or equal to the number of elements.
         *
         * yield: (uint) -> void
         */
        auto partition = [] (uint elements, uint num_parts, auto yield)
        {
            auto n = elements / num_parts;
            auto r = elements % num_parts;

            for (uint i = 0; i < num_parts; ++i)
            {
                yield(n + (i < r));
            }
        };

        /**
         * Divide an interval into non-overlapping contiguous sub-intervals.
         *
         * yield: (i0: uint, di: uint) -> void
         */
        auto subdivide = [partition] (uint a, uint da, uint num_parts, auto yield)
        {
            partition(da, num_parts, [yield, &a] (uint n)
            {
                yield(a, n);
                a += n;
            });
        };

        subdivide(i0[0], di[0], num_parts, [this, yield] (uint i0, uint di)
        {
            auto space = *this;
            space.i0[0] = i0;
            space.di[0] = di;
            yield(space);
        });
    }
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
