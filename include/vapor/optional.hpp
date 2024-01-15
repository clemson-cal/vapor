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
#include <cstdlib>
#include <cassert>
#include <limits>

namespace vapor {




/**
 * A maybe-type, can hold a value or be empty
 * 
 */
template <typename T>
struct optional_t
{
    HD bool has_value() const
    {
        return _has_value;
    }
    HD const T& get() const
    {
        return _val;
    }
    HD T& get()
    {
        return _val;
    }
    template <class F, typename U = std::invoke_result_t<F, T>>
    HD optional_t<U> map(F f) const
    {
        return _has_value ? optional_t<U>{f(_val), true} : optional_t<U>{U(), false};
    }
    T _val;
    bool _has_value;
};




/**
 * Construct a non-empty optional type
 */
template<typename T>
HD auto some(T val)
{
    return optional_t<T>{val, true};
}

/**
 * Construct an empty optional type
 */
template<typename T>
HD auto none()
{
    return optional_t<T>{T(), false};
}

} // namespace vapor
