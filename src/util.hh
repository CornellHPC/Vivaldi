#ifndef UTILS_HH
#define UTILS_HH

#include "mpi.h"

#ifdef CUDA
void wake_gpus(int myrank);
#endif

int square_grid_dim(MPI_Comm comm);

bool is_square_grid(MPI_Comm comm);

int tile_dim(MPI_Comm comm, int x);

#endif // UTILS_HH
