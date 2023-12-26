#include <stdexcept>
#include "app_parse.hpp"
#include "app_print.hpp"
#include "visit_struct.hpp"




struct config_t
{
    int num_zones = 100;
    double tfinal = 0.0;
    vapor::ivec_t<2> shape;
    vapor::dvec_t<5> priml;
};
VISITABLE_STRUCT(config_t, num_zones, tfinal, shape, priml);




void scan(const char* str, unsigned int, unsigned int& val)
{
    sscanf(str, "%u", &val);
}
void scan(const char* str, unsigned int, int& val)
{
    sscanf(str, "%d", &val);
}
void scan(const char* str, unsigned int, float& val)
{
    sscanf(str, "%f", &val);
}
void scan(const char* str, unsigned int, double& val)
{
    sscanf(str, "%lf", &val);
}




template<typename D, unsigned int S>
void scan(const char* str, unsigned int size, vapor::vec_t<D, S>& val)
{
    enum class lexer_state {
        ready,
        expect_sep,
        expect_end,
    };

    auto state = lexer_state::ready;
    unsigned int m = 0;

    for (unsigned int n = 0; n < size; ++n)
    {
        switch (state)
        {
        case lexer_state::ready:
            scan(&str[n], size - n, val[m]);
            m += 1;
            if (m == S)
                state = lexer_state::expect_end;
            else
                state = lexer_state::expect_sep;
            break;

        case lexer_state::expect_sep:
            if (str[n] == ',' || str[n] == ' ')
                state = lexer_state::ready;
            break;

        case lexer_state::expect_end:
            if (str[n] == ',')
                throw std::runtime_error("too many entries for vec");
            break;
        }
    }
    if (m < S) {
        throw std::runtime_error("too few entries for vec");
    }
}




int main(int argc, const char **argv)
{
    auto config = config_t();

    try {
        for (int n = 1; n < argc; ++n)
        {
            auto found = false;

            vapor::scan_key_val(argv[n], '\n', [&config, &found] (auto l, auto nl, auto r, auto nr)
            {
                visit_struct::for_each(config, [&found, l, nl, r, nr] (auto key, auto& val)
                {

                    if (strncmp(l, key, nl) == 0)
                    {
                        scan(r, nr, val);
                        found = true;
                    }                    
                });
                if (! found)
                {
                    throw std::runtime_error("key not found");
                }
            });
        }
        auto print_pair = [] (auto n, const auto& v)
        {
            vapor::print(n);
            vapor::print(": ");
            vapor::print(v);
            vapor::print("\n");
        };
        visit_struct::for_each(config, print_pair);
    }
    catch (std::exception& e)
    {
        printf("%s\n", e.what());
    }
}
