#define VAPOR_MPI
#include "vapor/comm.hpp"
#include "vapor/executor.hpp"
#include "vapor/print.hpp"

using namespace vapor;




void create_cartesian_communicator()
{
    auto comm = communicator_t<3>();
    auto space = index_space(uvec(30, 30, 30));

    if (comm.rank() == 0) {
        printf("create a 3d cartesian communicator...\n");
    }
    for (int i = 0; i < comm.size(); ++i)
    {
        if (i == comm.rank())
        {
            print(format("hello from rank %d/%d; ", i, comm.size()));
            print(comm.coords());
            print(" --- ");
            print(comm.subspace(space));
            print("\n");
        }
        comm.barrier();
    }
}

// The MPI global array feature is on hold for now. The header file comm.hpp
// still contains relevant support routines, but the 'global' member function
// in the array class has been removed, as well as the special case of global
// array in the cache method.
// 
// That code was removed after commit 669b415
// 
// 
// void create_global_array()
// {
//     auto comm = communicator_t<1>();

//     if (comm.rank() == 0) {
//         printf("create a 1d global array...\n");
//     }
//     auto exec = cpu_executor_t();
//     auto alloc = shared_ptr_allocator_t();
//     auto a = range(50).global(comm).cache(exec, alloc);

//     for (int rank = 0; rank < comm.size(); ++rank)
//     {
//         if (rank == comm.rank())
//         {
//             for (int i = a._subspace.i0[0]; i < a._subspace.i0[0] + a._subspace.di[0]; ++i)
//             {
//                 print(a[i]);
//                 print(" ");
//             }
//             print("| ");
//         }
//         comm.barrier();
//     }
//     if (comm.rank() == 0) {
//         print("\n");
//     }
// }

void create_mpi_datatype()
{
    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size != 2) {
        print("create_mpi_datatype requires exactly 2 procs\n");
        return;
    }
    if (rank == 0) {
        print("create an MPI data type for vec_t and send an instance...\n");
    }

    auto vec_type = mpi_repr<vec_t<double, 5>>::type();

    if (rank == 0)
    {
        auto a = zeros_vec<double, 5>();
        for (int i = 0; i < 5; ++i)
        {
            a[i] = i;
        }
        MPI_Send(&a.data, 1, vec_type, 1, 0, MPI_COMM_WORLD);
    }
    else
    {
        auto status = MPI_Status();
        auto a = zeros_vec<double, 5>();
        MPI_Recv(&a.data, 1, vec_type, 0, 0, MPI_COMM_WORLD, &status);
        print(a);
        print("\n");
    }
    MPI_Type_free(&vec_type);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        print("create an MPI data type for a subarray...\n");
    }
    auto subarray = mpi_subarray_datatype<vec_t<int, 3>>(index_space(uvec(100, 100)), index_space(uvec(10, 10)));
    MPI_Type_free(&subarray);
}

int main()
{
    auto mpi = mpi_scoped_initializer();
    create_cartesian_communicator();
    // create_global_array();
    create_mpi_datatype();
    return 0;
}
