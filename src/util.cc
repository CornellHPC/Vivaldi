#include <cmath>
#include <iostream>

#include "mpi.h"

#ifdef CUDA
#include <cuda_runtime.h>

void wake_gpus(int myrank) {
  int ndevices;
  cudaGetDeviceCount(&ndevices);
  if (myrank == 0) {
    std::cout << "Number of GPUs per node " << ndevices << "\n" << std::flush;
    std::cout << "Waking the GPUs..." << std::flush;
  }
  int ts = 0;
  for (int i = 0; i < ndevices; ++i) {
    cudaSetDevice(i);
    int *array;
    int *dArray;
    int count = 7;
    int size = count * sizeof(int);
    array = new int[count];
    for (int j = 0; j < count; j += 1)
      array[j] = j;
    cudaMalloc(&dArray, size);
    cudaMemcpy(dArray, array, size, cudaMemcpyHostToDevice);
    cudaFree(dArray);
    delete[] array;
  }
  if (myrank == 0)
    std::cout << " DONE!\n" << std::flush;
}
#endif

int square_grid_dim(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return std::floor(std::sqrt(size));
}

bool is_square_grid(MPI_Comm comm) {
  int size, sr;
  MPI_Comm_size(comm, &size);
  sr = square_grid_dim(comm);
  return sr * sr == size;
}

int tile_dim(MPI_Comm comm, int x) {
  int p = square_grid_dim(comm);
  return (x / p) + ((x % p == 0) ? 0 : 1);
}
