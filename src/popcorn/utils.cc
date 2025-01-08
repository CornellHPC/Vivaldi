#include <cmath>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>

#include "utils.hh"

void popcorn::wake_gpus(int rank) {
  int ndevices;
  cudaGetDeviceCount(&ndevices);

  if (rank == 0) {
    std::cout << "Number of GPUs per node: " << ndevices << "\n" << std::flush;
    std::cout << "Waking the GPUs..." << std::flush;
  }

  for (int i = 0; i < ndevices; ++i) {
    cudaSetDevice(i);
    int* array;
    int* dArray;
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

  if (rank == 0) std::cout << " DONE!\n" << std::flush;
}