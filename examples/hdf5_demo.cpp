#include "hdf5_repr.hpp"
#include "hdf5_native.hpp"
#include "hdf5_string.hpp"
#include "hdf5_vector.hpp"
#include "hdf5_array.hpp"
#include "app_print.hpp"




struct config_t
{
    int a;
    double b;
    bool c;
    std::string d;
    vapor::dvec_t<3> e;
    vapor::shared_array_t<1, double> f;
    std::vector<int> g;
};

VISITABLE_STRUCT(config_t, a, b, c, d, e, f, g);




int main()
{
    auto conf1 = config_t{
        5,
        2.3,
        true,
        "hey",
        {2.3, 3.1, 1.0},
        vapor::range(3).map([] (auto i) { return double(i); }).cache(),
        {0, 1, 2, 3, 4}};
    auto conf2 = config_t();

    {
        auto h5f = H5Fcreate("hdf5_demo.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        vapor::hdf5_write(h5f, "conf", conf1);
        H5Fclose(h5f);
    }

    {
        auto h5f = H5Fopen("hdf5_demo.h5", H5P_DEFAULT, H5P_DEFAULT);
        vapor::hdf5_read(h5f, "conf", conf2);
        H5Fclose(h5f);
    }

    auto print_pair = [] (auto n, const auto& v)
    {
        vapor::print(n);
        vapor::print(": ");
        vapor::print(v);
        vapor::print("\n");
    };
    visit_struct::for_each(conf1, print_pair); vapor::print("\n");
    visit_struct::for_each(conf2, print_pair);

    return 0;
}
