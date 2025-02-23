#include <cassert>

#include "mpi.h"

#include "cpop/cluster.hh"
#include "cpop/compute_kernel.hh"
#include "cpop/utils.hh"

using namespace cpop;

float EPSILON = 0.01;

template <typename T>
void assert_buffer_equal(T* b0, T* b1, int64_t count) {
  for (int i = 0; i < count; ++i) {
    assert(std::abs(b0[i] - b1[i]) <= EPSILON);
  }
}

void check_k(DnMat_t& K, int rank) {
  int64_t count = K.h_ * K.w_;
  float* m = (float*)malloc(count * sizeof(float));
  cudaMemcpy(m, K.dM, count * sizeof(float), cudaMemcpyDeviceToHost);

  if (rank == 0) {
    float buffer[22] = {205,  493,  493,  1293, 781,  2093, 1069, 2893,
                        1357, 3693, 1645, 4493, 1933, 5293, 2221, 6093,
                        2509, 6893, 2797, 7693, 3085, 8493};
    assert_buffer_equal(buffer, m, count);
  } else if (rank == 1) {
    float buffer[22] = {781,   1069,  2093,  2893,  3405,  4717,  4717, 6541,
                        6029,  8365,  7341,  10189, 8653,  12013, 9965, 13837,
                        11277, 15661, 12589, 17485, 13901, 19309};
    assert_buffer_equal(buffer, m, count);
  } else if (rank == 2) {
    float buffer[22] = {1357,  1645,  3693,  4493,  6029,  7341,  8365,  10189,
                        10701, 13037, 13037, 15885, 15373, 18733, 17709, 21581,
                        20045, 24429, 22381, 27277, 24717, 30125};
    assert_buffer_equal(buffer, m, count);
  } else if (rank == 3) {
    float buffer[55] = {1933,  2221,  2509,  2797,  3085,  5293,  6093,  6893,
                        7693,  8493,  8653,  9965,  11277, 12589, 13901, 12013,
                        13837, 15661, 17485, 19309, 15373, 17709, 20045, 22381,
                        24717, 18733, 21581, 24429, 27277, 30125, 22093, 25453,
                        28813, 32173, 35533, 25453, 29325, 33197, 37069, 40941,
                        28813, 33197, 37581, 41965, 46349, 32173, 37069, 41965,
                        46861, 51757, 35533, 40941, 46349, 51757, 57165};
    assert_buffer_equal(buffer, m, count);
  } else {
    assert("Unexpected rank.");
  }

  free(m);
}

void check_v0(V_t& V) {
  int64_t count = 11;
  int64_t* a = (int64_t*)malloc(count * sizeof(int64_t));
  cudaMemcpy(a, V.global_assignments, count * sizeof(int64_t),
             cudaMemcpyDeviceToHost);

  int64_t buffer[11] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  assert_buffer_equal(buffer, a, count);

  free(a);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  wake_gpus(rank);
  slate::gpu_aware_mpi(true);
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  int m = 11;
  int n = 8;
  int k = 2;

  auto PT = load_matrix("../data/small", m, n, comm);

  /** WARNING: Need to update t computation for different m and p */
  int t = (rank == size - 1) ? 5 : 2;
  int* t_sizes = (int*)calloc(size, sizeof(int));
  for (int i = 0; i < size; ++i) {
    t_sizes[i] = (i == size - 1) ? 5 : 2;
  }

  DnMat_t K(m, t, compute_kernel_matrix(PT));
  PT.releaseWorkspace();
  check_k(K, rank);

  V_t V(m, t, k, t_sizes, comm);
  check_v0(V);

  DnMat_t E(k, t);
  DnVec_t z(t);
  DnVec_t c(k);

  MPI_Finalize();
  return 0;
}
