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




#include <stdexcept>
#include "vapor/parse.hpp"
#include "vapor/print.hpp"
#include "visit_struct/visit_struct.hpp"




struct config_t
{
    int num_zones = 100;
    double tfinal = 0.0;
    bool cache = false;
    vapor::ivec_t<2> shape;
    vapor::dvec_t<5> left;
    vapor::dvec_t<5> right;
};
VISITABLE_STRUCT(config_t, num_zones, tfinal, cache, shape, left, right);




int main(int argc, const char **argv)
{
    auto config = config_t();

    try {
        vapor::set_from_key_vals(config, argc, argv);
        vapor::print(config);
    }
    catch (std::exception& e)
    {
        printf("%s\n", e.what());
    }
    return 0;
}
