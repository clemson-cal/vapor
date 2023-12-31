# VAPOR: Versatile Accelerated Physics Optimization Routines

## Description

Vapor is a lightweight application framework to ease the development of
GPU-accelerated and massively parallel scientific simulation codes. Its
objectives are to:

- Provide lightweight, idiomatic C++ abstractions for high-performance array
  transformations and PDE solvers
- Adapt to hybrid parallel compute architectures (multi-core / multi-node /
  multi-GPU) with zero or nearly zero change to the application code
- Enable rapid development of small, targeted, simulation codes, by
  abstracting the business logic of simulation drivers
- Be header-only and dependency-free, requires only a C++17 compiler is
  (optional dependencies include CUDA, MPI, and HDF5)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

## Installation

Todo...

## Usage

### Functional N-dimensional arrays

Let's start with an example. The following code creates an array of integers
`a`, and maps it to an array of doubles, `b`. Both of these arrays are purely
functional; they are not associated with a buffer of memory, but rather
generate values lazily when indexed. On the final line, the array `b` is
cached to a memory-backed array `c`. Indexing the elements of `c` 

```c++
auto a = vapor::range(100);                     /* a[10] == 10 */
auto b = a.map([] (int i) { return 2.0 * i; }); /* b[10] == 20.0 */
auto c = b.cache();                             /* c[10] == 20.0 */
```

#### Rationale

Numeric arrays in Vapor comprise a D-dimensional index space, and a function
`f: ivec_t<D> -> T`. Arrays are logically immutable; `a[i]` returns by value
an element of type `T`. This means that array assignment by indexing is not
possible, `a[i] = x` will not compile because the left-hand-side is not a
reference type. Arrays are transformed mainly by mapping operatons. If `g: T
-> U` then `a.map(g)` is an array with value type of `U`, and has the same
index space as `a`. Array elements are computed lazily, meaning that for the
array `b = a.map(f).map(g).map(h)`, each time `b[i]` appears in the code, the
function composition `h(g(f(i))` is executed.

An array can be cached to a "memory-backed array" by calling `a.cache()`. The
resulting memory-backed array uses strided memory access to retrieve the value
of a multi-dimensional index in the buffer.

Unlike arrays from other numeric libraries, including numpy, arrays in
Vapor can have a non-zero starting index. This changes the semantics of
inserting the values of one array into another, often for the better, and
is also favorable in dealing with global arrays and domain decomposition.
For example, if `a` covers the 1d index space (0, 100) and `b` covers (1, 99),
then the array `a.insert(b)` has the values of `a` at the endpoints, and the
values of `b` on the interior.

In-place array modifications are modeled using a paradigm inspired by Jax. If
a is an array, the construct `a = a.at(space).map(f)` will map only the
elements inside the index space through the function `f`, leaving the other
values unchanged.

#### Executors

Hardware-agnostic accelerated array transformations in Vapor are accomplished
by use of an "executor" construct. The execution of a lazily evaluated array,
to a memory-backed one, is done by parallelizing the traveral of the array,
either to multiple CPU cores, or to a "device" kernel in the case of GPU
executions. There is also a bare-bones sequential executor which sequentially
traverses arrays.

#### Allocators

Todo...

### Stack-allocated vectors

Vapor provides a stack-allocated 1d vector type `vec_t<T, S>`. `S` is an
unsigned integer which sets the compile-time size of the vector. The behavior
is not too different from `std::array`, except that `vec_t` can be compiled in
CUDA device code, and overloads the arithmetic operators.

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
