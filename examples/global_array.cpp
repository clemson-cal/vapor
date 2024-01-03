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

void create_global_array()
{
    auto comm = communicator_t<1>();

    if (comm.rank() == 0) {
        printf("create a 1d global array...\n");
    }
    auto exec = cpu_executor_t();
    auto alloc = shared_ptr_allocator_t();
    auto a = range(50).global(comm).cache(exec, alloc);

    for (int rank = 0; rank < comm.size(); ++rank)
    {
        if (rank == comm.rank())
        {
            for (int i = a._subspace.i0[0]; i < a._subspace.i0[0] + a._subspace.di[0]; ++i)
            {
                print(a[i]);
                print(" ");
            }
            print("| ");
        }
        comm.barrier();
    }
    if (comm.rank() == 0) {
        print("\n");
    }
}

int main()
{
    auto mpi = mpi_scoped_initializer();
    create_cartesian_communicator();
    create_global_array();
    return 0;
}
