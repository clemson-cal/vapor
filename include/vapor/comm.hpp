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
#ifdef VAPOR_MPI
#include <cassert>
#include <mpi.h>
#include "vec.hpp"

namespace vapor {




template<uint D>
class communicator_t
{
public:
    communicator_t(ivec_t<D> shape=zeros_ivec<D>(), ivec_t<D> topology=zeros_ivec<D>())
    {
        auto num_nodes = 0;
        auto reorder = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
        MPI_Dims_create(num_nodes, D, shape);
        MPI_Cart_create(MPI_COMM_WORLD, D, shape, topology, reorder, &_comm);
    }
    int size() const
    {
        int s;
        MPI_Comm_size(_comm, &s);
        return s;
    }
    int rank() const
    {
        int r;
        MPI_Comm_rank(_comm, &r);
        return r;
    }
    void barrier() const
    {
        MPI_Barrier(_comm);
    }
    ivec_t<D> coords() const
    {
        auto s = ivec_t<D>();
        auto t = ivec_t<D>();
        auto c = ivec_t<D>();
        MPI_Cart_get(_comm, D, s, t, c);
        return c;
    }
    ivec_t<D> shape() const
    {
        auto s = ivec_t<D>();
        auto t = ivec_t<D>();
        auto c = ivec_t<D>();
        MPI_Cart_get(_comm, D, s, t, c);
        return s;
    }
    ivec_t<D> topology() const
    {
        auto s = ivec_t<D>();
        auto t = ivec_t<D>();
        auto c = ivec_t<D>();
        MPI_Cart_get(_comm, D, s, t, c);
        return t;
    }
private:
    MPI_Comm _comm;
};

}

#endif // VAPOR_MPI
