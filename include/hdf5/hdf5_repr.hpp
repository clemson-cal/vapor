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
#include <hdf5.h>
#include "vapor/runtime.hpp"
#include "visit_struct/visit_struct.hpp"

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
 *     static hid_t space(const T&) { return H5Screate(H5S_SCALAR); }
 *     static hid_t type(const T&) { return H5Tcopy(...); }
 *     template<class A> static void allocate(T&, hid_t space, hid_t type, A& alloc) { }
 * }
 */
template<typename T> struct hdf5_repr;




/**
 * If std::map or something like it needs to be written to HDF5, this struct
 * can be specialized as
 *
 * template<typename U>
 * struct is_key_value_container_t<map<string, U>> : public true_type {};
 * 
 * This will allow the container to be written conveniently by calling e.g.
 * hdf5_write(h5_loc, "my_container", the_map).
 */
template <typename T>
struct is_key_value_container_t : public std::false_type {};




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
	auto group = hid_t();
        if (H5Lexists(location, name, H5P_DEFAULT))
            group = H5Gopen(location, name, H5P_DEFAULT);
        else
            group = H5Gcreate(location, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        visit_struct::for_each(val, [group] (const char *key, const auto& item)
        {
            hdf5_write(group, key, item);
        });
        H5Gclose(group);
    }
    else if constexpr (is_key_value_container_t<T>::value) {
	auto group = hid_t();
        if (H5Lexists(location, name, H5P_DEFAULT))
            group = H5Gopen(location, name, H5P_DEFAULT);
        else
            group = H5Gcreate(location, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (const auto& [key, item] : val) {
            hdf5_write(group, key.data(), item);
        }
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

template<typename T>
void hdf5_write_file(const char *filename, const T& val)
{
    auto h5f = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hdf5_write(h5f, "/", val);
    H5Fclose(h5f);
}




/**
 * Read an HDF5 representable object from an HDF5 location
 * 
 */
template<typename T, class A>
void hdf5_read(hid_t location, const char *name, T& val, A& alloc)
{
    struct op_data_t
    {
        T* value;
        A* alloc;
    };
    if constexpr (visit_struct::traits::is_visitable<T>::value) {
        auto group = H5Gopen(location, name, H5P_DEFAULT);
        visit_struct::for_each(val, [group, &alloc] (const char *key, auto& item)
        {
            hdf5_read(group, key, item, alloc);
        });
        H5Gclose(group);
    }
    else if constexpr (is_key_value_container_t<T>::value) {
        auto group = H5Gopen(location, name, H5P_DEFAULT);
        auto op_data = op_data_t{&val, &alloc};
        auto op = [] (hid_t group, const char *key, const H5L_info_t *info, void *op_data) -> herr_t {
            T& value = *((op_data_t*) op_data)->value;
            A& alloc = *((op_data_t*) op_data)->alloc;
            hdf5_read(group, key, value[key], alloc);
            return 0;
        };
        hsize_t idx = 0;
        H5Literate(group, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, op, &op_data);
        H5Gclose(group);
    }
    else {
        auto set = H5Dopen(location, name, H5P_DEFAULT);
        auto space = H5Dget_space(set);
        auto type = H5Dget_type(set);
        hdf5_repr<T>::allocate(val, space, type, alloc);
        H5Dread(set, type, space, space, H5P_DEFAULT, hdf5_repr<T>::data(val));
        H5Tclose(type);
        H5Sclose(space);
        H5Dclose(set);
    }
}

template<typename T>
void hdf5_read(hid_t location, const char *name, T& val)
{
    hdf5_read(location, name, val, Runtime::allocator());
}

template<typename T, class A>
void hdf5_read_file(const char *filename, T& val, A& alloc)
{
    auto h5f = H5Fopen(filename, H5P_DEFAULT, H5P_DEFAULT);
    hdf5_read(h5f, "/", val, alloc);
    H5Fclose(h5f);
}

template<typename T>
void hdf5_read_file(const char *filename, T& val)
{
    hdf5_read_file(filename, val, Runtime::allocator());
}

} // namespace vapor
