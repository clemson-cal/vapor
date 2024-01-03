#define VAPOR_MPI
#include "vapor/comm.hpp"
#include "vapor/print.hpp"




int main()
{
    auto mpi = vapor::mpi_scoped_initializer();
    auto comm = vapor::communicator_t<3>();
    auto space = vapor::index_space(vapor::uvec(30, 30, 30));

    for (int i = 0; i < comm.size(); ++i)
    {
        if (i == comm.rank())
        {
            vapor::print(vapor::format("hello from rank %d/%d; ", i, comm.size()));
            vapor::print(comm.coords());
            vapor::print(" --- ");
            vapor::print(comm.subspace(space));
            vapor::print("\n");
        }
        comm.barrier();
    }
    return 0;
}
