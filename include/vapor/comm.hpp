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
#include "array.hpp"
#include "index_space.hpp"
#include "vec.hpp"

namespace vapor {




/**
 * A template struct to explain how a type is represented to MPI
 *
 */
template<typename T> struct mpi_repr;




template<> struct mpi_repr<signed char>        { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_SIGNED_CHAR, &t); return t; } };
template<> struct mpi_repr<unsigned char>      { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_UNSIGNED_CHAR, &t); return t; } };
template<> struct mpi_repr<short>              { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_SHORT, &t); return t; } };
template<> struct mpi_repr<unsigned short>     { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_UNSIGNED_SHORT, &t); return t; } };
template<> struct mpi_repr<int>                { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_INT, &t); return t; } };
template<> struct mpi_repr<unsigned int>       { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_UNSIGNED, &t); return t; } };
template<> struct mpi_repr<long>               { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_LONG, &t); return t; } };
template<> struct mpi_repr<unsigned long>      { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_UNSIGNED_LONG, &t); return t; } };
template<> struct mpi_repr<long long>          { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_LONG_LONG, &t); return t; } };
template<> struct mpi_repr<unsigned long long> { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_UNSIGNED_LONG_LONG, &t); return t; } };
template<> struct mpi_repr<char>               { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_CHAR, &t); return t; } };
template<> struct mpi_repr<wchar_t>            { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_WCHAR, &t); return t; } };
template<> struct mpi_repr<float>              { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_FLOAT, &t); return t; } };
template<> struct mpi_repr<double>             { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_DOUBLE, &t); return t; } };
template<> struct mpi_repr<long double>        { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_LONG_DOUBLE, &t); return t; } };
template<> struct mpi_repr<bool>               { static MPI_Datatype type() { MPI_Datatype t; MPI_Type_dup(MPI_C_BOOL, &t); return t; } };

template<typename U, uint S>
struct mpi_repr<vec_t<U, S>>
{
    static MPI_Datatype type()
    {
        MPI_Datatype u = mpi_repr<U>::type();
        MPI_Datatype s;
        MPI_Type_contiguous(S, u, &s);
        MPI_Type_commit(&s);
        MPI_Type_free(&u);
        return s;
    }
};

template<typename U, uint D>
MPI_Datatype mpi_subarray_datatype(index_space_t<D> parent, index_space_t<D> nested)
{
    MPI_Datatype u = mpi_repr<U>::type();
    MPI_Datatype s;
    ivec_t<D> sizes;
    ivec_t<D> subsizes;
    ivec_t<D> starts;
    int order = MPI_ORDER_C;

    for (int n = 0; n < D; ++n)
    {
        sizes[n] = parent.di[n];
        subsizes[n] = nested.di[n];
        starts[n] = nested.i0[n];
    }
    MPI_Type_create_subarray(D, sizes, subsizes, starts, order, u, &s);
    MPI_Type_commit(&s);
    MPI_Type_free(&u);
    return s;
}




class mpi_scoped_initializer
{
public:
    mpi_scoped_initializer()
    {
        MPI_Init(0, nullptr);
    }
    ~mpi_scoped_initializer()
    {
        MPI_Finalize();
    }
    int size() const
    {
        int s;
        MPI_Comm_size(MPI_COMM_WORLD, &s);
        return s;
    }
    int rank() const
    {
        int r;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
    }
    void barrier() const
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }
};




template<uint D>
class communicator_t
{
public:
    communicator_t(ivec_t<D> shape=zeros_ivec<D>(), ivec_t<D> topology=ones_ivec<D>())
    {
        auto num_nodes = 0;
        auto reorder = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
        MPI_Dims_create(num_nodes, D, shape);
        MPI_Cart_create(MPI_COMM_WORLD, D, shape, topology, reorder, &_comm);
    }
    ~communicator_t()
    {
        MPI_Comm_free(&_comm);
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
    index_space_t<D> subspace(index_space_t<D> space) const
    {
        auto c = coords();
        auto s = shape();

        for (uint axis = 0; axis < D; ++axis)
        {
            space = space.subspace(s[axis], c[axis], axis);
        }
        return space;
    }
    template<class F, class E, class A>
    auto expand(const array_t<D, F>& a, int count, E& executor, A& allocator)
    {
        using T = typename array_t<D, F>::value_type;
        auto is_exp = a.space().expand(count);
        auto is_ori = is_exp.with_start(zeros_uvec<D>());
        auto result = zeros<T>(is_exp).insert(a).cache(executor, allocator);

        for (int axis = 0; axis < D; ++axis)
        {
            int sendrank;
            int recvrank;
            int sendcount = 1;
            int recvcount = 1;
            MPI_Datatype sendtype_r = mpi_subarray_datatype<T>(is_ori, is_ori.upper(count, axis).shift(-count, axis));
            MPI_Datatype recvtype_r = mpi_subarray_datatype<T>(is_ori, is_ori.lower(count, axis));
            MPI_Datatype sendtype_l = mpi_subarray_datatype<T>(is_ori, is_ori.lower(count, axis).shift(+count, axis));
            MPI_Datatype recvtype_l = mpi_subarray_datatype<T>(is_ori, is_ori.upper(count, axis));
            MPI_Status status;
            MPI_Cart_shift(_comm, axis, +1, &recvrank, &sendrank);
            MPI_Sendrecv(result._data, sendcount, sendtype_r, sendrank, 0,
                         result._data, recvcount, recvtype_r, recvrank, 0,
                         _comm, &status);
            MPI_Cart_shift(_comm, axis, -1, &recvrank, &sendrank);
            MPI_Sendrecv(result._data, sendcount, sendtype_l, sendrank, 0,
                         result._data, recvcount, recvtype_l, recvrank, 0,
                         _comm, &status);
            MPI_Type_free(&sendtype_l);
            MPI_Type_free(&recvtype_l);
            MPI_Type_free(&sendtype_r);
            MPI_Type_free(&recvtype_r);
        }
        return result;
    }

private:
    MPI_Comm _comm;
};

}

#endif // VAPOR_MPI
