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
    template<class A> static void allocate(T& val, hid_t space, hid_t type, A&)
    {
        val.resize(H5Tget_size(type));
    }
};

} // namespace vapor
