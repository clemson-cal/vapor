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
    static hid_t space(const T& val)
    {
        hsize_t dims[1] = { val.size() };
        return H5Screate_simple(1, dims, nullptr);
    }
    static hid_t type(const T& val)
    {
        return hdf5_repr<U>::type(U());
    }
    template<class A> static void allocate(T& val, hid_t space, hid_t type, A&)
    {
        val.resize(H5Sget_simple_extent_npoints(space));
    }
};

} // namespace vapor
