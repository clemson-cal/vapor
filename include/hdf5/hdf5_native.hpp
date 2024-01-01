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
#include "hdf5_repr.hpp"

namespace vapor {




/**
 * HDF5 representation of native data types
 * 
 */
#define HDF5_REPR_POD(T, h)                                        \
template<> struct hdf5_repr<T>                                     \
{                                                                  \
    static const T* data(const T& val) { return &val; }            \
    static T* data(T& val) { return &val; }                        \
    static void allocate(T&, hid_t space, hid_t type) { }          \
    static hid_t space(const T&) { return H5Screate(H5S_SCALAR); } \
    static hid_t type(const T&) { return H5Tcopy(h); }             \
}
HDF5_REPR_POD(char, H5T_NATIVE_CHAR);
HDF5_REPR_POD(signed char, H5T_NATIVE_SCHAR);
HDF5_REPR_POD(unsigned char, H5T_NATIVE_UCHAR);
HDF5_REPR_POD(short, H5T_NATIVE_SHORT);
HDF5_REPR_POD(unsigned short, H5T_NATIVE_USHORT);
HDF5_REPR_POD(int, H5T_NATIVE_INT);
HDF5_REPR_POD(unsigned int, H5T_NATIVE_UINT);
HDF5_REPR_POD(long, H5T_NATIVE_LONG);
HDF5_REPR_POD(unsigned long, H5T_NATIVE_ULONG);
HDF5_REPR_POD(long long, H5T_NATIVE_LLONG);
HDF5_REPR_POD(unsigned long long, H5T_NATIVE_ULLONG);
HDF5_REPR_POD(float, H5T_NATIVE_FLOAT);
HDF5_REPR_POD(double, H5T_NATIVE_DOUBLE);
HDF5_REPR_POD(long double, H5T_NATIVE_LDOUBLE);




/**
 * Representation of C++ bool type in HDF5, compatible with h5py
 */
template<> struct hdf5_repr<bool>
{
    using T = bool;
    static const T* data(const T& val) { return &val; }
    static T* data(T& val) { return &val; }
    static void allocate(T&, hid_t space, hid_t type) { }
    static hid_t space(const T&) { return H5Screate(H5S_SCALAR); }
    static hid_t type(const T&)
    {
        auto t = true;
        auto f = false;
        auto type = H5Tcreate(H5T_ENUM, sizeof(bool));
        H5Tenum_insert(type, "TRUE", &t);
        H5Tenum_insert(type, "FALSE", &f);
        return type;
    }
};

} // namespace vapor
