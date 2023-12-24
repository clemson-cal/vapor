#pragma once
#include "core_compat.hpp"
#include "core_vec.hpp"

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
