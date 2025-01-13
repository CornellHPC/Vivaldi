#include "utils.hh"

void popcorn::wake_gpus(int rank) {
  int ndevices;
  cudaGetDeviceCount(&ndevices);

  if (rank == 0) {
    std::cout << "Number of GPUs per node: " << ndevices << "\n" << std::flush;
    std::cout << "Waking the GPUs..." << std::flush;
  }

  for (int i = 0; i < ndevices; ++i)
    cudaSetDevice(i);

  if (rank == 0)
    std::cout << " DONE!\n" << std::flush;
}

void popcorn::print_device_buffer(float* buf, size_t count, int rank_to_print) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == rank_to_print) {
    float* temp = (float*)malloc(count * sizeof(float));
    cudaMemcpy(temp, buf, count * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; i++)
      std::cout << temp[i] << " ";
    std::cout << std::endl;

    free(temp);
  }
}

int64_t popcorn::get_time_elapsed(
    std::chrono::_V2::system_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now() - start)
      .count();
}