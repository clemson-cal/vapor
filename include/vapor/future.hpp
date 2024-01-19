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


Use cases for executors:

- Load data for a device buffer: returns immediately, with the array itself
- Load data for a managed buffer: returns immediately, with a future
- Load data for a managed buffer, and accumulate a counter
*/

#pragma once

namespace vapor {
namespace future {




template <class F> auto future(F);




template <class F>
struct future_t
{
    using value_type = std::invoke_result_t<F>;
    auto get() const
    {
        return f();
    }
    // template <class G>
    // auto map(G g) const
    // {
    //     if constexpr (std::is_void_v<value_type>) {
    //         return future([*this, g] () { f(); return g(); });            
    //     } else {
    //         return future([*this, g] () { return g(f()); });
    //     }
    // }
    F f;
};

template <class F>
auto future(F f)
{
    return future_t<F>{f};
}

struct ready_t
{
    void operator()() const
    {
    }
};

static inline auto ready()
{
    return future(ready_t{});
}

template<typename T>
struct just_t
{
    auto operator()() const
    {
        return val;
    }
    T val;
};

template <typename T>
auto just(T val)
{
    return future(just_t<T>{val});
}

} // namespace future
} // namespace vapor
