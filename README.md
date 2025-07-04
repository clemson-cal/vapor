# VAPOR: __Versatile Accelerated Physics Optimization Routines__
## Description
Vapor aims to define a minimal set of C++ programming primitives needed to model
PDE solvers, and execute them on massively parallel compute hardware.

Vapor encourages a functional programming style, and facilitates concise
and expressive C++ code. For example, the following code implements a naive
method to solve a 2d heat equation:

```c++
auto N = uint(100);
auto s = index_space(vec(N, N));
auto i = indices(s);
auto h = 1.0 / N; /* grid spacing */
auto x = i.map([h] HD (vec_t<int, 2> ij) { return ij * h - 0.5; });
auto u = x.map([] HD (vec_t<double, 2> x) { return exp(-dot(x, x)); });
auto del_squared_u = i[s.contract(1)].map([u, h] HD (vec_t<int, 2> ij) {
    auto i = ij[0];
    auto j = ij[1];
    return (u[vec(i + 1, j)] +
            u[vec(i, j + 1)] +
            u[vec(i - 1, j)] +
            u[vec(i, j - 1)] - 4 * u[ij]) / (h * h); /* Laplacian, Del^2(u) */
});
auto dt = h * 0.1;
auto du = del_squared_u * dt;
auto u_next = (u + du).cache();
```
This code is "deploy-anywhere", in the sense that it can be compiled for execution
on a CPU or a GPU, if one is available. Vapor will take advantage of as many CPU
cores, or GPU devices, are available on your system. With minor changes, it can
also utilize CPU or GPU resources on distributed compute nodes.

Separation of the algorithm (e.g. the above code), from the details of how it
is parallelized, enables rapid, robust development of high-performance
numerical code. Compiler-optimized Vapor programs have zero runtime overhead,
and short compile times.

Vapor is also a lightweight application framework that can ease the development
of GPU-accelerated and massively parallel scientific simulation codes.


### Objectives
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
- [Library Usage](#library-usage)
- [Build System](#build-system)
- [Credits](#credits)
- [License](#license)


## Installation
The following will compile the programs in the examples directory, other than
the ones which require CUDA.

```bash
git clone https://github.com/clemson-cal/vapor
cd vapor
./configure
make
bin/array_demo_dbg # runs a single-core CPU debug version of the array demo program
```

If you are on a system with a CUDA-enabled GPU, the following will build and
run the array demo, this time enabled for GPU execution.

```bash
./configure --enable-cuda
make bin/array_demo_gpu
bin/array_demo_gpu
```


## Library Usage
### Functional N-dimensional arrays
Let's start with an example. The following code creates an array of integers
`a`, and maps it to an array of doubles, `b`. Both of these arrays
are _lazy_; they are not associated with a buffer of memory, but rather
generate values from a function when indexed. On the final line, the array
`b` is cached to a memory-backed array `c`. The elements of `c` are equal to
those of `b`, but are loaded from a memory buffer instead of using lazy
evaluation.

```c++
auto a = vapor::range(100);                     /* a[10] == 10 */
auto b = a.map([] (int i) { return 2.0 * i; }); /* b[10] == 20.0 */
auto c = b.cache();                             /* c[10] == 20.0 */
```

All computations are carried out in the final step where the array is cached,
and automatically parallelized either on CPU's or GPU's depending on the
build configuration. Hybrid parallelism is also possible by explicitly
passing an _executor_ instance to the `cache` function.


#### Executors
Hardware-agnostic accelerated array transformations in Vapor are accomplished
by use of an "executor" pattern. The executor is used to convert a
lazily-evaluated array to a memory-backed one. Vapor provides three basic
executor types, illustrated below, but can be easily extended with custom
executors.

```c++
auto pool = pool_allocator_t(); /* a memory pool, reuses buffer allocations */
auto cpu = cpu_executor_t(); /* single-core executor */
auto omp = omp_executor_t(); /* multi-core executor */
auto gpu = gpu_executor_t(4); /* multi-GPU executor (uses 4 devices) */
auto a = my_array.cache(cpu, pool);
auto b = my_array.cache(omp, pool);
auto c = my_array.cache(gpu, pool);
assert(all(a == b) && all(b == c));
```


#### Arrays in Vapor
Numeric arrays in Vapor are simply a pair of objects: a D-dimensional index
space, and a function `f: ivec_t<D> -> T`. Arrays are logically immutable; `a
[i]` returns by value an element of type `T`. This means that array
assignment by indexing is not possible, `a[i] = x` will not compile because
the left-hand-side is not a reference type. Arrays are transformed mainly by
mapping operatons. If `g: T -> U` then `a.map(g)` is an array with value type
of `U`, and has the same index space as `a`. Array elements are computed
lazily, meaning that for the array `b = a.map(f).map(g).map(h)`, each time `b
[i]` appears in the code, the function composition `h(g(f(i))` is executed.

An array can be cached to a "memory-backed array" by calling `a.cache()`. The
resulting memory-backed array uses strided memory access to retrieve the value
of a multi-dimensional index in the buffer.

Unlike arrays from other numeric libraries such as numpy, arrays in Vapor can
have a non-zero starting index. This simplifies the semantics of inserting
the values of one array into another, and is often favorable in dealing with
global arrays, domain decomposition, and stencil calculations. For example,
if `a` covers the 1d index space (0, 100) and `b` covers (1, 99), then the
array `a.insert(b)` has the values of `a` at the endpoints, and the values of
`b` on the interior. Array subsets are created by indexing an array like
this: `a[subspace]` where for example, `subspace = a.space().contract(2)`. If
`a` had a starting index of 0, then the array `a[subspace]` would have a
starting index of 2.

In-place array modifications are modeled using a paradigm inspired by Jax. If
`a` is an array, the construct `a = a.at(space).map(f)` will map only the
elements inside the index space through the function `f`, leaving the other
values unchanged. The array `a.at(space).map(f)` has the same index space as
`a`. It is also valid to add an array `du`, whose index space lies inside the
index space of another array `u`, using the syntax `u.add(du)`. The index
spaces of `u.add(du)`, and `u.mul(a)`, are both equal to `u.space()`. Note
that `u + du` generates a failed assertion unless `u` and `du` have the same
index spaces.


#### Modeling exceptions

In modeling of PDEs it's not uncommon for some calculations to be fallible.
For example, if a root finder is applied to each element of an array, the
root finder could fail to converge to a solution for some elements. In a
sequential computing environment (single-core CPU), crash reporting and
mitigation can be easily managed using exceptions, but exceptions are not
supported in GPU kernels, and they must be handled with care in a
multi-threaded environment.

For this reason, Vapor provides a facility to catch errors in kernel
executions, based on the use of an `optional` type. If the execution of a
certain function might fail, then its return type should be wrapped in an
optional type, for example,

```c++
vapor::optional_t<double> logarithm(double x)
{
    if (x > 0.0) {
        return vapor::some(log(x));
    } else {
        return vapor::none<double>();
    }
}
```

When an array of optional type is cached, it can optionally be
"unwrapped" at the same time, and a `std::runtime_error` exception will be
thrown if any of the array elements were `none` rather than `some`:

```c++
try {
    auto a = vapor::range(10).map([] (int i) { return 8.0 - i; }).map(logarithm);
    auto b = a.cache_unwrap(); // b[i] has type double
    // use result
} catch (const std::exception& e) {
    vapor::print(e.what());
    // enter crash mitigation, or exit gracefully
}
```


#### Index spaces
Todo...


#### Allocators
Todo...


### Stack-allocated vectors
Vapor provides a stack-allocated 1d vector type `vec_t<T, S>`. `S` is an
unsigned integer which sets the compile-time size of the vector. The behavior
is not too different from `std::array`, except that `vec_t` can be compiled in
CUDA device code, and overloads the arithmetic operators.


## Build System
### Scheme
Projects based on Vapor contains a collection of _programs_. A program refers
to a single `.cpp` file, designed for a specfic type of calculation.
That .cpp file includes the vapor header files, and should not have any
object file dependancies (yet, this could be relaxed in the future). A given
program can be compiled in any of several _modes_. The currently supported
modes are

- dbg: target compiled with `-O0 -g -D VAPOR_DEBUG`
- cpu: target compiled with `-Ofast`; CPU (single-core) executor is the default
- omp: target compiled with `omp_flags`; OMP executor is default
- gpu: target compiled with `nvcc`; GPU executor is default

This configure script can be run listing any of the above modes modes after
the `--modes` flag. One Makefile _target_ is written for each program, and
for each of the requested modes.

For example, the Vapor project contains an examples directory with a source
file called `array_demo.cpp`. If run with `--modes omp gpu`, then two targets
will be derived from the `array_demo` program: `array_demo_omp`, and
`array_demo_gpu`. The associated executables are placed in a bin/ directory
by default.


### Examples
`./configure`: Generates a Makefile with rules to build targets using a
default set of modes, or otherwise the modes named in the `system.json`
file.

`./configure --enable-cuda`: Creates targets also with gpu modes

`./configure --stdout`: Dumps the Makefile to the terminal


### Project file
The configure script looks in the current working directory for a JSON file
called `project.json`. An example project file can be found
[here](project.json) in the Vapor root directory.


### System file
The configure script optionally loads a JSON file called `system.json` from the
current directory. Allowed keys include:

- `libs`: Extra linker arguments which may be needed on a given system
- `omp_flags`: Equivalent of `-fopenmp`; defaults to `-Xpreprocessor -fopenmp`,
  which works on Darwin


## Credits
This library was authored by Jonathan Zrake over the 2023 - 2024 Winter break,
with significant intellectual contributions from [Marcus DuPont](https://github.com/EigenDev),
and Andrew MacFadyen.


## License
MIT License

Copyright (c) 2023-2024 Clemson Computational Astrophysics Lab

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
