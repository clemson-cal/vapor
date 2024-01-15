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




#define VAPOR_STD_STRING
#define VAPOR_STD_VECTOR
#define VAPOR_STD_MAP
#define VAPOR_USE_SHARED_PTR_ALLOCATOR
#include "vapor/print.hpp"
#include "vapor/executor.hpp"
#include "hdf5/hdf5_array.hpp"
#include "hdf5/hdf5_map.hpp"
#include "hdf5/hdf5_native.hpp"
#include "hdf5/hdf5_repr.hpp"
#include "hdf5/hdf5_string.hpp"
#include "hdf5/hdf5_vector.hpp"




struct config_t
{
    int a;
    double b;
    bool c;
    std::string d;
    vapor::dvec_t<3> e;
    vapor::memory_backed_array_t<1, float, std::shared_ptr> f;
    std::vector<int> g;
};
VISITABLE_STRUCT(config_t, a, b, c, d, e, f, g);




int main(int argc, const char **argv)
{
    auto conf1 = config_t{
        5,
        2.3,
        true,
        "hey",
        {2.3, 3.1, 1.0},
        vapor::range(6).map([] HD (int i) { return float(i); }).cache(),
        {0, 1, 2, 3, 4}
    };
    auto conf2 = config_t();

    vapor::hdf5_write_file("hdf5_demo.h5", conf1);
    vapor::hdf5_read_file("hdf5_demo.h5", conf2);
    vapor::print(conf1);
    vapor::print("\n");
    vapor::print(conf2);

    auto dict1 = std::map<std::string, double>();
    auto dict2 = std::map<std::string, double>();
    dict1["a"] = 1.0;
    dict1["b"] = 2.0;
    vapor::hdf5_write_file("hdf5_demo_map.h5", dict1);
    vapor::hdf5_read_file("hdf5_demo_map.h5", dict2);

    vapor::print(dict1);
    vapor::print("\n");
    vapor::print(dict2);
    vapor::print("\n");

    return 0;
}
