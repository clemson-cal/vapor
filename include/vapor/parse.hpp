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
#ifdef VAPOR_STD_VECTOR
#include <vector>
#endif
#ifdef VAPOR_STD_STRING
#include <string>
#endif
#include <cstring>
#include <stdexcept>
#include "print.hpp"
#include "vec.hpp"
#include "visit_struct/visit_struct.hpp"

namespace vapor {




/**
 * Invoke a parser callback on key-value pairs found in a string
 *
 * Valid key-value pairs include a = b, a= b, a =b, and a=b. Pairs are
 * separated by the character sep. Any space after the first non-whitespace
 * character in a value is included as a part of the value.
 *
 * The parser is invoked as parser(l, nl, r, nr) --- where l and r are char
 * pointers and nl and nr are the number of characters in the key or value.
 */
template<typename F>
static void scan_key_val(const char *str, char sep, F parser)
{
    enum class lexer_state {
        ready,
        lhs,
        expect_equals,
        expect_rhs,
        rhs,
        comment,
    };

    if (str == nullptr) {
        return;
    }

    auto state = lexer_state::ready;
    const char *lhs_start = nullptr;
    const char *lhs_final = nullptr;
    const char *rhs_start = nullptr;
    const char *rhs_final = nullptr;

    while (true)
    {
        char c = *str;

        switch (state)
        {
        case lexer_state::ready:
            if (isspace(c)) {
            }
            else if (isalpha(c) || c == '\n' || c == '\0') {
                state = lexer_state::lhs;
                // printf("ready -> lhs\n");
                lhs_start = str;
            }
            else if (c == '#') {
                state = lexer_state::comment;
                // printf("ready -> comment\n");
            }
            else {
                throw std::runtime_error(format("parse error: scan_key_val got bad identifier %s", str));
            }
            break;

        case lexer_state::lhs:
            if (c == '=') {
                state = lexer_state::expect_rhs;
                // printf("lhs -> expect_lhs\n");
                lhs_final = str;
            }
            else if (isspace(c)) {
                state = lexer_state::expect_equals;
                // printf("lhs -> expect_equals\n");
                lhs_final = str;
            }
            else if (c == '\0') {
                throw std::runtime_error("parse error: ot line ended without '='");
            }
            break;

        case lexer_state::expect_equals:
            if (isspace(c)) {
            }
            else if (c == '=') {
                state = lexer_state::expect_rhs;
                // printf("expect_equals -> expect_rhs\n");
            }
            else {
                throw std::runtime_error("parse error: ot line ended without '='");
            }
            break;

        case lexer_state::expect_rhs:
            if (c == '#' || c == sep || c == '\0') {
                // empty RHS calls parser with an empty string
                state = lexer_state::ready;
                rhs_start = str;
                rhs_final = str;
                parser(lhs_start, lhs_final - lhs_start,
                       rhs_start, rhs_final - rhs_start);
            }
            else if (! isspace(c)) {
                state = lexer_state::rhs;
                // printf("expect_rhs -> rhs\n");
                rhs_start = str;
            }
            break;

        case lexer_state::rhs:
            if (c == '#') {
                state = lexer_state::comment;
                // printf("rhs -> comment\n");
                rhs_final = str;
                parser(lhs_start, lhs_final - lhs_start,
                       rhs_start, rhs_final - rhs_start);
            }
            else if (c == sep || c == '\0') {
                state = lexer_state::ready;
                // printf("rhs -> ready\n");
                rhs_final = str;
                parser(lhs_start, lhs_final - lhs_start,
                       rhs_start, rhs_final - rhs_start);
            }
            break;

        case lexer_state::comment:
            if (c == '\n') {
                state = lexer_state::ready;
                // printf("comment -> ready\n");
            }
            break;
        }

        if (c == '\0') {
            return;
        }
        else {
            ++str;
        }
    }
}




void scan(const char* str, uint size, bool& val)
{
    if (size == 4 && strncmp(str, "true", 4) == 0) {
        val = true;
    }
    else if (size == 5 && strncmp(str, "false", 5) == 0) {
        val = false;
    }
    else {
        throw std::runtime_error(format("parse error: expected true|false, got %.*s", size, str));
    }
}
void scan(const char* str, uint, uint& val)
{
    sscanf(str, "%u", &val);
}
void scan(const char* str, uint, int& val)
{
    sscanf(str, "%d", &val);
}
void scan(const char* str, uint, float& val)
{
    sscanf(str, "%f", &val);
}
void scan(const char* str, uint, double& val)
{
    sscanf(str, "%lf", &val);
}




template<typename D, uint S>
void scan(const char* str, uint size, vec_t<D, S>& val)
{
    enum class lexer_state {
        ready,
        expect_sep,
        expect_end,
    };

    auto state = lexer_state::ready;
    uint m = 0;

    for (uint n = 0; n < size; ++n)
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
                throw std::runtime_error(format("parse error: vec size must be %d", S));
            break;
        }
    }
    if (m < S) {
        throw std::runtime_error(format("parse error: vec size must be %d", S));
    }
}




#ifdef VAPOR_STD_VECTOR
template<typename T>
void scan(const char* str, uint size, std::vector<T>& val)
{
    enum class lexer_state {
        ready,
        expect_sep_or_end,
    };
    auto state = lexer_state::ready;
    val.clear();

    for (uint n = 0; n < size; ++n)
    {
        switch (state)
        {
        case lexer_state::ready:
            val.emplace_back();
            scan(&str[n], size - n, val.back());
            state = lexer_state::expect_sep_or_end;
            break;
        case lexer_state::expect_sep_or_end:
            if (str[n] == ',' || str[n] == ' ')
                state = lexer_state::ready;
            break;
        }
    }
}
#endif // VAPOR_STD_VECTOR




#ifdef VAPOR_STD_STRING
static inline void scan(const char* str, uint size, std::string& val)
{
    val = std::string(str, size);
}
#endif // VAPOR_STD_STRING




template<typename T, typename = std::enable_if_t<visit_struct::traits::is_visitable<T>::value>>
auto set_from_key_vals(T& target, const char *str)
{
    auto found = false;

    scan_key_val(str, '\n', [&target, &found] (const char* l, size_t nl, const char* r, size_t nr)
    {
        visit_struct::for_each(target, [l, nl, r, nr, &found] (auto key, auto& val)
        {
            auto size = nl > strlen(key) ? nl : strlen(key);

            if (strncmp(l, key, size) == 0)
            {
                scan(r, nr, val);
                found = true;
            }                    
        });
        if (! found)
        {
            throw std::runtime_error(format("parse error: no data member '%.*s'", int(nl), l));
        }
    });
}




template<typename T, typename = std::enable_if_t<visit_struct::traits::is_visitable<T>::value>>
auto set_from_key_vals(T& target, int argc, const char **argv)
{
    for (int n = 1; n < argc; ++n)
    {
        if (argv[n][0] != '-') {
            set_from_key_vals(target, argv[n]);
        }
    }
}

} // namespace vapor
