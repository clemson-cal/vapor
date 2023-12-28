#pragma once
#include <vector>
#include "hdf5_repr.hpp"

namespace vapor {




/**
 * HDF5 representation of std::vector<U>
 * 
 */
template<typename U> struct hdf5_repr<std::vector<U>>
{
    using T = std::vector<U>;
    static const U* data(const T& val)
    {
        return val.data();
    }
    static U* data(T& val)
    {
        return val.data();
    }
    static void allocate(T& val, hid_t space, hid_t type)
    {
        val.resize(H5Sget_simple_extent_npoints(space));
    }
    static hid_t space(const T& val)
    {
        hsize_t dims[1] = { val.size() };
        return H5Screate_simple(1, dims, nullptr);
    }
    static hid_t type(const T& val)
    {
        return hdf5_repr<U>::type(U());
    }
};

} // namespace vapor
