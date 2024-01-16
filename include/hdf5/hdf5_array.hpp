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
    template <class A> static void allocate(T& val, hid_t space, hid_t type, A&)
    {
    }
};




/**
 * HDF5 representation of vapor::memory_backed_array_t
 * 
 * WARNING: Arrays with index spaces not starting at the origin should not be
 * written to HDF5, because the start index is not (yet) written to the HDF5
 * file.
 */
template<uint D, typename U, template<typename> typename P>
struct hdf5_repr<memory_backed_array_t<D, U, P>>
{
    using T = memory_backed_array_t<D, U, P>;
    static const U* data(const T& val)
    {
        return val.data();
    }
    static U* data(T& val)
    {
        return val.data();
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
    template<class A> static void allocate(T& val, hid_t dspace, hid_t type, A& allocator)
    {
        if (H5Sget_simple_extent_ndims(dspace) != D)
        {
            throw std::runtime_error("array in file has wrong number of dimensions");
        }
        hsize_t hdims[D];
        H5Sget_simple_extent_dims(dspace, hdims, nullptr);
        auto shape = uvec_t<D>{};

        for (uint n = 0; n < D; ++n)
        {
            shape[n] = hdims[n];
        }
        auto space = index_space(shape);
        auto buffer = allocator.allocate(space.size() * sizeof(U));
        auto table = lookup<U>(space, buffer);
        val = array(table, space, buffer.get());
    }
};

} // namespace vapor
