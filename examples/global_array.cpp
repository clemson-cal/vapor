#define VAPOR_MPI
#include "vapor/comm.hpp"
#include "vapor/print.hpp"




int main()
{
    MPI_Init(0, nullptr);

    auto comm = vapor::communicator_t<3>();

    for (int i = 0; i < comm.size(); ++i)
    {
        if (i == comm.rank())
        {
            vapor::print(vapor::format("hello from rank %d/%d; ", i, comm.size()));
            vapor::print(comm.coords());
            vapor::print("\n");
        }
        comm.barrier();
    }
    MPI_Finalize();
    return 0;
}
