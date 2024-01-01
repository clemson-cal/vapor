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
#include <memory>
#include "vapor/array.hpp"
#include "vapor/functional.hpp"
#include "vapor/memory.hpp"
#include "hdf5_repr.hpp"

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
template<uint D, typename U> struct hdf5_repr<memory_backed_array_t<D, U, std::shared_ptr>>
{
    using T = memory_backed_array_t<D, U, std::shared_ptr>;
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
        auto shape = uvec_t<D>{};

        for (uint n = 0; n < D; ++n)
        {
            shape[n] = hdims[n];
        }
        auto start = zeros_uvec<D>();
        auto stride = strides_row_major(shape);
        auto memory = std::make_shared<vapor::managed_memory_t>(product(shape) * sizeof(U));
        auto data = (U*) memory->data();
        auto table = lookup(start, stride, data, memory);
        val = array(table, index_space(start, shape), data);
    }
    static hid_t space(const T& val)
    {
        hsize_t dims[D];

        for (uint n = 0; n < D; ++n)
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
