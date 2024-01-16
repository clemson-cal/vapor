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

#include "compat.hpp"

namespace vapor {

template <class G>
struct future_t
{
    using value_type = std::invoke_result_t<G>;
    value_type get() const
    {
        return g();
    }
    G g;
};

namespace future {

template <class G>
auto future(G g)
{
    return future_t<G>{g};
}

static inline auto ready()
{
    return future([] {});
}

template <typename T>
auto just(T val)
{
    return future([val] () { return val; });
}

#ifdef __CUDACC__
auto device_synchronize(int device)
{
    return future([device] () {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    });
}
#endif


// template<typename T>
// struct immediate_future_t
// {
//     using value_type = T;
//     value_type get() const
//     {
//         return val;
//     }
//     T val;
// };


// template<typename T>
// struct read_from_device_buffer_future_t
// {
//     using value_type = T;
//     value_type get() const
//     {
//         return buffer_ptr->template read<T>();
//     }
//     ref_counted_ptr_t<buffer_t> buffer_ptr;
//     int device;
// };


// template<class F, class G>
// struct mapped_future_t
// {
//     using value_type = std::invoke_result_t<G, typename F::value_type>;
//     value_type get() const
//     {
//         return function(upstream.get());
//     }
//     F upstream;
//     G function;
// };

} // namespace future
} // namespace vapor
