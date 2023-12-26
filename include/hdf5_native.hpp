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
        H5Tenum_insert(type, "FALSE", &f);
        H5Tenum_insert(type, "TRUE", &t);
        return type;
    }
};

} // namespace vapor
