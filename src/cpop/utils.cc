#include <iostream>

#include "utils.hh"

namespace cpop {

void wake_gpus(int rank) {
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

int64_t get_time_elapsed(std::chrono::_V2::system_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now() - start)
      .count();
}

}  // namespace cpop
