#pragma once
#include <string>
#include "hdf5_repr.hpp"

namespace vapor {




/**
 * HDF5 representation of std::string
 * 
 */
template<> struct hdf5_repr<std::string>
{
    using T = std::string;
    static const char* data(const T& val)
    {
        return val.data();
    }
    static char* data(T& val)
    {
        return val.data();
    }
    static void allocate(T& val, hid_t space, hid_t type)
    {
        val.resize(H5Tget_size(type));
    }
    static hid_t space(const T&)
    {
        return H5Screate(H5S_SCALAR);
    }
    static hid_t type(const T& val)
    {
        auto type = H5Tcopy(H5T_C_S1);
        H5Tset_size(type, val.size());
        return type;
    }
};

} // namespace vapor
