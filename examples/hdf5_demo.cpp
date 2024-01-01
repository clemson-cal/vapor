#define VAPOR_STD_STRING
#define VAPOR_STD_VECTOR
#define VAPOR_STD_MAP
#include "app_print.hpp"
#include "core_executor.hpp"
#include "hdf5_array.hpp"
#include "hdf5_map.hpp"
#include "hdf5_native.hpp"
#include "hdf5_repr.hpp"
#include "hdf5_string.hpp"
#include "hdf5_vector.hpp"




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
    auto exec = vapor::cpu_executor_t();
    auto alloc = vapor::shared_ptr_allocator_t();
    auto conf1 = config_t{
        5,
        2.3,
        true,
        "hey",
        {2.3, 3.1, 1.0},
        vapor::range(6).map([] (auto i) { return float(i); }).cache(exec, alloc),
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
