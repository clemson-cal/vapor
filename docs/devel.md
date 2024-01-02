# TODO
[-] thread pool executor (performs worse than OpenMP; see ideas/thread_pool_executor.cpp)
[x] restore separate executor-allocator paradigm for array cache's
[-] use of executor base class (not possible due to template members)
[ ] output directory feature for simulation driver
[ ] diagnostic and time series outputs
[ ] crash / safety mode feature
[x] simulation base class
[x] array reductions
[ ] parse std::vector in parse.hpp
[ ] rename header files to remove core, app; place in include/vapor
[x] iterate properly over HDF5 links on key-val container read
[x] project organization

# Rationale for MPI-global arrays
The idea of a global array facility would be trivial if not for stencil
operations. A functionally generated array has an index space that is much too
big to fit any one node. When the array is cached, the executor creates an MPI
cartesian communicator, and returns another array with the same (very large)
index space, but a memory backing for only the local portion of the array.
Subsequent mapping operations over the partially memory-backed array are
applied only to the local index space. This can be accomplished with the
simply addition of a "local_space" data member to the array struct, reflecting
the portion of the array which can be operated on by the local process.

When stencil operations are involved, only a selection of the array is
targeted to be updated by a set or map opperation. The complement of the
targeted array portion must be filled either by a boundary condition or by
exchanging data through the cartesian commmunicator.

```c++
u = (u.at(interior) + du).cache();
```

In the current library dessign, the object returned by u.at(interior) + du is
a regular array, one that returns the new value of u, u + du, inside the
targeted index space (interior), and the old value otherwise. If instead, yet
another new type of object were returned, say an "incomplete array" or
something, which remembered the array selection that had last been updated,
that information could be used in exchanging guard zone data following the
subsequent cache operation. Unfortunately that approach could fail if several
updates are to be chained, for example

```c++
auto u1 = u0.at(interior) + du;
auto u2 = u1.at(left_side).set(bc_values);
auto u3 = u2.cache();
```

Now there were two recent targeted updates, and only one of them (the first)
should be followed by a communication step. This suggests the communication
step cannot be implied as one might have hoped by a preceding targeted update,
and perhaps moreover, that the communication step should not be implicit at
all.

