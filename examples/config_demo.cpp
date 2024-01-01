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
