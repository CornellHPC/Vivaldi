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

  if (rank == 0)
    std::cout << " DONE!\n" << std::flush;
}

void popcorn::print_device_buffer(float* buf, size_t count) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  float* temp = (float*)malloc(count * sizeof(float));
  cudaMemcpy(temp, buf, count * sizeof(float), cudaMemcpyDeviceToHost);

  if (rank == 0) {
    for (int i = 0; i < count; i++)
      std::cout << temp[i] << " ";

    std::cout << std::endl;
  }

  free(temp);
}