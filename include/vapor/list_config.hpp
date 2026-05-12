/**
 * Vapor list_config_field: file-backed name list config type.
 *
 * Stores a user-provided spec string ("@file.txt" or inline
 * names) and the resolved contents in fixed-size char buffers.
 * Round-trips through HDF5 as a visitable-struct group with two
 * char-array datasets. Avoids std::string HDF5 path entirely.
 */
#pragma once

#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vapor/parse.hpp"
#include "vapor/print.hpp"
#include "vapor/vec.hpp"
#include "visit_struct/visit_struct.hpp"

namespace vapor {

struct list_config_field
{
    static constexpr uint capacity = 1024;

    // Not visitable; used only for error message prefixing.
    const char* key = "list_config_field";

    // Visitable: persisted to HDF5 as char-array datasets.
    vec_t<char, capacity> spec     = {};
    vec_t<char, capacity> resolved = {};

    list_config_field(const char* key_ = "list_config_field") : key(key_) {}

    bool empty() const
    {
        return trim(to_string(spec)).empty();
    }

    std::string raw_spec() const
    {
        return to_string(spec);
    }

    std::string resolved_text() const
    {
        return to_string(resolved);
    }

    // Resolve the resolved_text against an iterable of const char*
    // (e.g., a static const char* names[N] or std::array<const char*, N>).
    // Returns the set of indices whose names match tokens in resolved_text.
    // Token "all" alone selects every entry. Mixed "all"+other -> throw.
    // Unknown name -> throw.
    template<class NameTable>
    std::set<uint> resolve(const NameTable& names) const
    {
        auto tokens = tokenize(resolved_text());
        auto cols   = std::set<uint>();

        if (tokens.empty()) {
            return cols;
        }

        auto n_all = uint(0);
        for (const auto& t : tokens) {
            if (t == "all") n_all += 1;
        }
        if (n_all != 0 && tokens.size() != 1) {
            throw std::runtime_error(format(
                "parse error: %s: token 'all' cannot be mixed with explicit names",
                key));
        }
        if (n_all == 1) {
            uint col = 0;
            for (auto it = std::begin(names); it != std::end(names); ++it, ++col) {
                if (*it != nullptr) cols.insert(col);
            }
            return cols;
        }
        for (const auto& tok : tokens) {
            auto found = false;
            uint col = 0;
            for (auto it = std::begin(names); it != std::end(names); ++it, ++col) {
                if (*it != nullptr && tok == *it) {
                    cols.insert(col);
                    found = true;
                    break;
                }
            }
            if (! found) {
                throw std::runtime_error(format(
                    "parse error: %s: unknown name '%s'",
                    key, tok.c_str()));
            }
        }
        return cols;
    }

    void assign(std::string input)
    {
        input = trim(input);
        put(spec, input, "spec");

        auto text = input;

        if (! input.empty() && input[0] == '@') {
            auto path = input.substr(1);
            if (path.empty()) {
                throw std::runtime_error(format(
                    "parse error: %s: empty @file path", key));
            }
            if (path.find_first_of(" \t\r\n,") != std::string::npos) {
                throw std::runtime_error(format(
                    "parse error: %s: @file path contains whitespace or comma: '%s'",
                    key, path.c_str()));
            }
            auto file = std::ifstream(path);
            if (! file) {
                throw std::runtime_error(format(
                    "parse error: %s: could not open @file '%s'",
                    key, path.c_str()));
            }
            auto buf = std::ostringstream();
            buf << file.rdbuf();
            text = buf.str();

            // Reject recursive @ includes early via substring scan,
            // before tokenization (avoids reading huge adversarial
            // recursive files).
            if (text.find('@') != std::string::npos) {
                for (const auto& tok : tokenize(text)) {
                    if (! tok.empty() && tok[0] == '@') {
                        throw std::runtime_error(format(
                            "parse error: %s: recursive @ include not supported in '%s'",
                            key, path.c_str()));
                    }
                }
            }
        }
        put(resolved, text, "resolved");
    }

private:
    static std::string to_string(const vec_t<char, capacity>& v)
    {
        auto n = size_t(0);
        while (n < capacity && v.data[n] != '\0') ++n;
        return std::string(v.data, n);
    }

    static void put(vec_t<char, capacity>& dst, const std::string& src, const char* role)
    {
        if (src.size() >= capacity) {
            throw std::runtime_error(format(
                "parse error: list_config_field %s exceeds %u bytes",
                role, capacity - 1));
        }
        for (uint i = 0; i < capacity; ++i) dst[i] = '\0';
        std::memcpy(dst.data, src.data(), src.size());
    }

    static std::string trim(const std::string& s)
    {
        auto b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return {};
        auto e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    }

    // Tokenize: whitespace and comma are separators. '#' starts a
    // line comment to end of line. Returns words in order.
    static std::vector<std::string> tokenize(const std::string& text)
    {
        auto clean = std::string();
        auto in_comment = false;
        for (char c : text) {
            if (in_comment) {
                if (c == '\n') { in_comment = false; clean.push_back(' '); }
                continue;
            }
            if (c == '#') { in_comment = true; clean.push_back(' '); }
            else if (c == ',') clean.push_back(' ');
            else clean.push_back(c);
        }
        auto tokens = std::vector<std::string>();
        auto stream = std::istringstream(clean);
        auto word   = std::string();
        while (stream >> word) tokens.push_back(word);
        return tokens;
    }
};

} // namespace vapor

VISITABLE_STRUCT(vapor::list_config_field, spec, resolved);

namespace vapor {

static inline void scan(const char* str, uint size, list_config_field& val)
{
    val.assign(std::string(str, size));
}

static inline void print(const list_config_field& val, FILE* file = stdout)
{
    print(val.raw_spec(), file);
}

} // namespace vapor
