#pragma once
#include "hdf5_repr.hpp"
#include "core_array.hpp"

namespace vapor {




/**
 * HDF5 representation of vapor::vec_t<U, S>
 * 
 */
template<typename U, uint S> struct hdf5_repr<vapor::vec_t<U, S>>
{
    using T = vapor::vec_t<U, S>;
    static const U* data(const T& val)
    {
        return val.data;
    }
    static U* data(T& val)
    {
        return val.data;
    }
    static void allocate(T& val, hid_t space, hid_t type)
    {
    }
    static hid_t space(const T&)
    {
        return H5Screate(H5S_SCALAR);
    }
    static hid_t type(const T& val)
    {
        hsize_t dims[1] = { S };
        auto element_type = hdf5_repr<U>::type(U());
        auto type = H5Tarray_create(element_type, 1, dims);
        H5Tclose(element_type);
        return type;
    }
};




/**
 * HDF5 representation of vapor::shared_array_t<D, U>
 * 
 */
template<uint D, typename U> struct hdf5_repr<vapor::shared_array_t<D, U>>
{
    using T = vapor::shared_array_t<D, U>;
    static const U* data(const T& val)
    {
        return val._data;
    }
    static U* data(T& val)
    {
        return val._data;
    }
    static void allocate(T& val, hid_t space, hid_t type)
    {
        if (H5Sget_simple_extent_ndims(space) != D)
        {
            throw std::runtime_error("array in file has wrong number of dimensions");
        }
        hsize_t hdims[D];
        H5Sget_simple_extent_dims(space, hdims, nullptr);
        auto shape = vapor::uvec_t<D>{};

        for (int n = 0; n < D; ++n)
        {
            shape[n] = hdims[n];
        }
        auto start = vapor::zeros_uvec<D>();
        auto stride = vapor::strides_row_major(shape);
        auto memory = std::make_shared<vapor::managed_memory_t>();
        auto data = memory->allocate<U>(product(shape));
        auto lookup = vapor::lookup(start, stride, data, memory);
        val = array(lookup, index_space(start, shape), data);
    }
    static hid_t space(const T& val)
    {
        hsize_t dims[D];

        for (int n = 0; n < D; ++n)
        {
            dims[n] = val._shape[n];
        }
        return H5Screate_simple(D, dims, nullptr);
    }
    static hid_t type(const T& val)
    {
        return hdf5_repr<U>::type(U());
    }
};

} // namespace vapor
