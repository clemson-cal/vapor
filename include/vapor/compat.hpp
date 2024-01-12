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

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

namespace vapor {
	using uint = unsigned int;
}

// Enables bounds-checking on vapor arrays
// #define VAPOR_ARRAY_BOUNDS_CHECK

// Enables bounds-checking on vapor vecs
// #define VAPOR_VEC_BOUNDS_CHECK

// Enables utility functions on std::map
// #define VAPOR_STD_MAP

// Enables utility functions on std::string
// #define VAPOR_STD_STRING

// Enables utility functions on std::vector
// #define VAPOR_STD_VECTOR

// Sets the default allocator to one that allocates using std::shared_ptr of
// mananged memory blocks. When this macro is not defined, the default
// allocator is the pool allocator. Note that the shared pointer allocator is
// not compatible with CUDA, and may be slower than the pool allocator due to
// heavy use of system malloc. The shared pointer allocator is thread-safe
// whereas the pool allocator is not.
// 
// #define VAPOR_USE_SHARED_PTR_ALLOCATOR

// This could be readily increased if ever needed, it simply enables static
// allocations of per-device data in a few places.
// 
#define VAPOR_MAX_DEVICES 8
