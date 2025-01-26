#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cusparse.h"
#include "mpi.h"

#include "cpop/compute_c.hh"
#include "cpop/compute_kernel.hh"
#include "cpop/compute_sparse.hh"
#include "cpop/cusparse_helpers.hh"
#include "cpop/utils.hh"

using namespace cpop;

using hrc = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

/**
 * Runs the distributed popcorn kernel k-means clustering algorithm.
 * Usage: srun cpop [path] [m] [n] [k]
 *
 * @param path path to the dataset
 * @param m number of samples in the dataset
 * @param n number of features
 * @param k number of clusters to form
 */
int main(int argc, char* argv[]) {
  /** INITIALIZE MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  /** PARSE ARGUMENTS */
  assert(argc == 5 && "Invalid args. Must provide params [path] [m] [n] [k]");
  char* fpath = argv[1];
  int m = std::atoi(argv[2]);
  int n = std::atoi(argv[3]);
  int k = std::atoi(argv[4]);

  /** INITIALIZE GPU */
  wake_gpus(rank);
  slate::gpu_aware_mpi(false);
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  /** LOAD DATA */
  auto PT = load_matrix(fpath, m, n, comm);
  // slate::print("PT", PT);

  /** START THE TIMER (after dataset IO) */
  auto start = hrc::now();

  /** COMPUTE K */
  auto k_start = hrc::now();
  auto K_loc = compute_kernel_matrix(PT);
  PT.releaseWorkspace();
  auto k_elapsed = get_time_elapsed(k_start);
  // print(K_loc, rank);

  /** INITIALIZE V */
  auto vi_start = hrc::now();
  cusparseSpMatDescr_t gV;
  cusparseSpMatDescr_t lV;
  int t = rank == size - 1 ? m / size + m % size : m / size;
  int* t_sizes = (int*)calloc(size, sizeof(int));
  for (int i = 0; i < size; ++i)
    t_sizes[i] = i == size - 1 ? m / size + m % size : m / size;
  init_V(&gV, &lV, m, t, k, t_sizes, comm);
  auto vi_elapsed = get_time_elapsed(vi_start);

  /** Allocate variables for K-means loop */
  cusparseDnMatDescr_t ET;
  init_ET(gV, K_loc, &ET);
  // TODO: assert that t equals the width of ET and K_loc
  cusparseDnVecDescr_t z, c;
  init_z(t, &z);
  init_c(k, &c);

  /** K MEANS CLUSTERING LOOP */
  int niter = 1;
  for (int i = 0; i < niter; ++i) {
    spmm(handle, gV, K_loc, ET);  // SoMM: ET = VK using global V
    compute_z(lV, ET, z);         // Calculate z from the mask of local V on ET
    spmv(handle, lV, z, c);       // SpMV: c = Vz using local V
  }

  /** PRINT TIMES */
  MPI_Barrier(comm);
  if (rank == 0) {
    std::cout << "Time K: " << k_elapsed << "ms" << std::endl;
    std::cout << "Time Init V: " << vi_elapsed << "ms" << std::endl;
  }

  /** DESTROY */
  destroy(K_loc);
  destroy(gV);
  destroy(lV);
  destroy(ET);
  destroy(z);
  destroy(c);
  cusparseDestroy(handle);

  /** EXIT */
  MPI_Finalize();
  return 0;
}
