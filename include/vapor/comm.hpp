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




/**
 * Return an MPI data type for a vec_t
 *
 * The data type returned by this function needs to be closed with
 * MPI_Comm_free.
 * 
 */
template<typename U, uint S>
struct mpi_repr<vec_t<U, S>>
{
    static MPI_Datatype type()
    {
        auto u = mpi_repr<U>::type();
        auto s = MPI_Datatype();
        MPI_Type_contiguous(S, u, &s);
        MPI_Type_commit(&s);
        MPI_Type_free(&u);
        return s;
    }
};




/**
 * Return an MPI data type for a subset of an array
 *
 * The data type returned by this function corresponds to a sub-array with the
 * nested index space, within the parent index space. The data type needs to
 * be closed with MPI_Comm_free.
 * 
 */
template<typename U, uint D>
MPI_Datatype mpi_subarray(index_space_t<D> parent, index_space_t<D> nested)
{
    assert(parent.contains(nested));
    auto u = mpi_repr<U>::type();
    auto s = MPI_Datatype();
    auto sizes = cast<int>(parent.shape());
    auto subsizes = cast<int>(nested.shape());
    auto starts = cast<int>(nested.start());
    auto order = MPI_ORDER_C;
    MPI_Type_create_subarray(D, sizes, subsizes, starts, order, u, &s);
    MPI_Type_commit(&s);
    MPI_Type_free(&u);
    return s;
}




/**
 * An RAII-type scoped MPI initializer
 *
 * int main()
 * {
 *     auto mpi = mpi_scoped_initialize_t(); // calls MPI_Init
 *
 *     print(mpi.rank()); // convenience function to get size and rank
 *
 *     return 0; // MPI_Finalize called in the destructor
 * }
 * 
 */
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




/**
 * A wrapper for an MPI cartesian communicator, with convenience functions
 *
 * For example, instances can return a local index subspace on the current MPI
 * process, within a global space, and can add halo zones to a
 * multi-dimensional array by communicating with neighbor processes.
 */
template<uint D>
class cartesian_communicator_t
{
public:
    cartesian_communicator_t(ivec_t<D> shape=zeros_ivec<D>(), ivec_t<D> topology=ones_ivec<D>())
    {
        auto num_nodes = 0;
        auto reorder = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
        MPI_Dims_create(num_nodes, D, shape);
        MPI_Cart_create(MPI_COMM_WORLD, D, shape, topology, reorder, &_comm);
    }
    ~cartesian_communicator_t()
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
    uvec_t<D> coords(int rank) const
    {
        auto c = ivec_t<D>();
        MPI_Cart_coords(_comm, rank, D, c);
        return cast<uint>(c);
    }
    uvec_t<D> coords() const
    {
        auto s = ivec_t<D>();
        auto t = ivec_t<D>();
        auto c = ivec_t<D>();
        MPI_Cart_get(_comm, D, s, t, c);
        return cast<uint>(c);
    }
    uvec_t<D> shape() const
    {
        auto s = ivec_t<D>();
        auto t = ivec_t<D>();
        auto c = ivec_t<D>();
        MPI_Cart_get(_comm, D, s, t, c);
        return cast<uint>(s);
    }
    uvec_t<D> topology() const
    {
        auto s = ivec_t<D>();
        auto t = ivec_t<D>();
        auto c = ivec_t<D>();
        MPI_Cart_get(_comm, D, s, t, c);
        return cast<uint>(t);
    }

    /**
     * Return the local index space, within a global one, for any MPI process
     *
     */
    index_space_t<D> subspace(index_space_t<D> space, int rank) const
    {
        return space.subspace(shape(), coords(rank));
    }

    /**
     * Return the local index space, within a global one, for this MPI process
     *
     */
    index_space_t<D> subspace(index_space_t<D> space) const
    {
        return space.subspace(shape(), coords());
    }

    /**
     * Fill halo zones in a memory-backed array with data from neighbor ranks
     *
     * This function receives the number of zones (count) to be filled on each
     * side of the array, and performs a send-recv operation to fill those
     * zones. The of the input array is not changed.
     * 
     */
    template<typename T, template<typename> typename P>
    auto fill_halo(memory_backed_array_t<D, T, P>& a, uvec_t<D> count)
    {
        auto s = a.space().with_start(zeros_ivec<D>());

        for (int axis = 0; axis < D; ++axis)
        {
            auto n = int();
            auto m = int();
            auto sendtype_r = mpi_subarray<T>(s, s.upper(count[axis], axis).shift(-count[axis], axis));
            auto sendtype_l = mpi_subarray<T>(s, s.lower(count[axis], axis).shift(+count[axis], axis));
            auto recvtype_r = mpi_subarray<T>(s, s.lower(count[axis], axis));
            auto recvtype_l = mpi_subarray<T>(s, s.upper(count[axis], axis));
            auto status = MPI_Status();
            MPI_Cart_shift(_comm, axis, +1, &n, &m);
            MPI_Sendrecv(a._data, 1, sendtype_r, m, 0, a._data, 1, recvtype_r, n, 0, _comm, &status);
            MPI_Sendrecv(a._data, 1, sendtype_l, n, 0, a._data, 1, recvtype_l, m, 0, _comm, &status);
            MPI_Type_free(&sendtype_l);
            MPI_Type_free(&recvtype_l);
            MPI_Type_free(&sendtype_r);
            MPI_Type_free(&recvtype_r);
        }
    }

    /**
     * Return an array expanded with halo zones from neighbor MPI processes
     *
     * This function receives the number of zones (count) to be added to each
     * side of the array, and performs a send-recv operation to fill those
     * zones. The shape of the resulting array is a.shape() + 2 * count.
     * 
     */
    template<class F, class E, class A>
    auto expand(const array_t<D, F>& a, uvec_t<D> count, E& executor, A& allocator)
    {
        using T = typename array_t<D, F>::value_type;
        auto result = zeros<T>(a.space().expand(count)).insert(a).cache(executor, allocator);
        fill_halo(result, count);
        return result;
    }

    /**
     * Reconstruct a distributed array on a single process, root
     *
     * The index space of the array argument to this function must match the
     * result of the subspace
     */
    template<class F, class E, class A>
    auto reconstruct(array_t<D, F>& a, index_space_t<D> global_space, E& executor, A& allocator, int root=0)
    {
        assert(subspace(global_space) == a.space());
        using T = typename array_t<D, F>::value_type;
        auto gs = global_space;
        auto result = zeros<T>(gs).insert(a).cache(executor, allocator);

        if (rank() == root)
        {
            for (int r = 0; r < size(); ++r)
            {
                if (r == root)
                {
                    continue;
                }
                auto status = MPI_Status();
                auto recvtype = mpi_subarray<T>(gs, subspace(gs, r));
                MPI_Recv(result._data, 1, recvtype, r, 0, _comm, &status);
                MPI_Type_free(&recvtype);
            }
        }
        else
        {
            auto sendtype = mpi_subarray<T>(gs, a.space());
            MPI_Send(result._data, 1, sendtype, root, 0, _comm);
            MPI_Type_free(&sendtype);
        }
        return result;
    }

private:
    MPI_Comm _comm;
};

}

#endif // VAPOR_MPI
