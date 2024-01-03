# VAPOR: Versatile Accelerated Physics Optimization Routines

## Description

VAPOR is a lightweight application framework to ease the development of
GPU-accelerated and massively parallel scientific simulation codes. Its
objectives are to:

- Provide lightweight, idiomatic C++ abstractions for high-performance array
  transformations and PDE solvers
- Adapt to hybrid parallel compute architectures (multi-core / multi-node /
  multi-GPU) with near-zero change to the application code
- Enable rapid development of small, targeted, simulation codes, by
  abstracting the business logic of simulation drivers; user configurations,
  checkpointing, and real-time post-processing are done automatically by
  inheriting `Simulation` abstract base class
- Be header-only and dependency-free; a C++17 compiler is sufficient (optional
  dependencies include CUDA, MPI, and HDF5)

## Table of Contents (Optional)

If your README is long, add a table of contents to make it easy for users to find what they need.

- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

## Installation

Todo...

## Usage

### Functional N-dimensional arrays

Numeric arrays in VAPOR are defined as a D-dimensional index space, together
with a function `f: uvec_t<D> -> T`. Arrays are logically immutable, `a[i]`
returns by value an element of type `T`; `a[i] = x` will not compile. Arrays
are transformed mainly by mapping operatons. If `g: T -> U` then `a.map(g)`
is an array with value type of `U`, and has the same index space as a. Array
elements are computed lazily, meaning that `b = a.map(f).map(g).map(h)`
triggers the execution `h(g(f(i))` each time `b[i]` appears.

An array is cached to a "memory-backed array" by calling `a.cache(exec,
alloc)`, where `exec` is an executor and `alloc` is an allocator. The executor
can be a GPU executor on GPU-enabled platforms, or a multi-core executor
where OpenMP is available. The memory backed array uses strided memory
access to retrieve the value of a multi-dimensional index in the buffer.

Unlike arrays from other numeric libraries, including numpy, arrays in
VAPOR can have a non-zero starting index. This changes the semantics of
inserting the values of one array into another, often for the better, and
is also favorable in dealing with global arrays and domain decomposition.
For example, if `a` covers the 1d index space (0, 100) and `b` covers (1, 99),
then the array `a.insert(b)` has the values of `a` at the endpoints, and the
values of `b` on the interior.

In-place array modifications are modeled using a paradigm inspired by Jax.
If a is an array, the construct `a = a.at(space).map(f)` will map only the
elements inside the index space through the function `f`, leaving the other
values unchanged.

### Stack-allocated vectors

VAPOR provides a stack-allocated 1d vector type `vec_t<T, S>`. `S` is an
unsigned integer which sets the compile-time size of the vector. The
behavior is not too different from `std::array`, except that `vec_t` can
be compiled in CUDA device code, and overloads the arithmetic operators.

## Credits

Todo...

## License

MIT License

Copyright (c) 2023 Clemson Computational Astrophysics Lab

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
