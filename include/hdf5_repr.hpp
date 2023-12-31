#pragma once
#include "hdf5.h"
#include "visit_struct.hpp"

namespace vapor {




/**
 * A template struct to explain how a type is represented to HDF5
 *
 * A template specialization to represent a POD type T to HDF5 looks something
 * like this:
 *
 * 
 * template<> struct hdf5_repr<T>
 * {
 *     static const T* data(const T& val) { return &val; }
 *     static T* data(T& val) { return &val; }
 *     static void allocate(T&, hid_t space, hid_t type) { }
 *     static hid_t space(const T&) { return H5Screate(H5S_SCALAR); }
 *     static hid_t type(const T&) { return H5Tcopy(...); }
 * }
 */
template<typename T> struct hdf5_repr;




/**
 * Write an HDF5 representable object to an HDF5 location
 *
 * Rationale: visitable data structures are represented as HDF5 groups, with
 * one dataset or group per data member (data members may themselves be
 * visitable, resulting in a nested hierarchy).
 *
 * Data types which are not visitable are represented as data sets. The type
 * parameter U in a container object, (e.g. std::vector<U>), can thus not be
 * visitable.
 * 
 */
template<typename T>
void hdf5_write(hid_t location, const char *name, const T& val)
{
    if constexpr (visit_struct::traits::is_visitable<T>::value) {
        auto group = H5Gcreate(location, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        visit_struct::for_each(val, [group] (const char *name, const auto& val)
        {
            hdf5_write(group, name, val);
        });
        H5Gclose(group);
    }
    else {
        auto type = hdf5_repr<T>::type(val);
        auto space = hdf5_repr<T>::space(val);
        auto set = H5Dcreate(location, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(set, type, space, space, H5P_DEFAULT, hdf5_repr<T>::data(val));
        H5Dclose(set);
        H5Sclose(space);
        H5Tclose(type);
    }
}




/**
 * Read an HDF5 representable object from an HDF5 location
 * 
 */
template<typename T>
void hdf5_read(hid_t location, const char *name, T& val)
{
    if constexpr (visit_struct::traits::is_visitable<T>::value) {
        auto group = H5Gopen(location, name, H5P_DEFAULT);
        visit_struct::for_each(val, [group] (const char *name, auto& val)
        {
            hdf5_read(group, name, val);
        });
        H5Gclose(group);
    }
    else {
        auto set = H5Dopen(location, name, H5P_DEFAULT);
        auto space = H5Dget_space(set);
        auto type = H5Dget_type(set);
        hdf5_repr<T>::allocate(val, space, type);
        H5Dread(set, type, space, space, H5P_DEFAULT, hdf5_repr<T>::data(val));
        H5Tclose(type);
        H5Sclose(space);
        H5Dclose(set);
    }
}




template<typename T>
void hdf5_write_file(const char *filename, const T& val)
{
    auto h5f = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    visit_struct::for_each(val, [h5f] (const char *name, const auto& val)
    {
        hdf5_write(h5f, name, val);
    });
    H5Fclose(h5f);
}

template<typename T>
void hdf5_read_file(const char *filename, T& val)
{
    auto h5f = H5Fopen(filename, H5P_DEFAULT, H5P_DEFAULT);
    visit_struct::for_each(val, [h5f] (const char *name, auto& val)
    {
        hdf5_read(h5f, name, val);
    });
    H5Fclose(h5f);
}

} // namespace vapor
