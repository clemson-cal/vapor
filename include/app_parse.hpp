#pragma once
#include <stdexcept>

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
                throw std::runtime_error("[ready] bad identifier");
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
                throw std::runtime_error("[lhs] line ended without '='");
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
                throw std::runtime_error("[expect_equals] line ended without '='");
            }
            break;

        case lexer_state::expect_rhs:
            if (c == '#' || c == sep || c == '\0') {
                throw std::runtime_error("[expect_rhs] expected a value");
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

} // namespace vapor
