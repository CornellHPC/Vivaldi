#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "cusparse.h"
#include "mpi.h"

#include "cpop/cluster.hh"
#include "cpop/compute_kernel.hh"
#include "cpop/utils.hh"

using namespace cpop;

using hrc = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

/**
 * The distributed popcorn kernel k-means clustering algorithm.
 *
 * @param path path to the dataset
 * @param m number of samples in the dataset
 * @param n number of features
 * @param k number of clusters to form
 */
int cluster(char* path, int m, int n, int k, MPI_Comm comm) {
  /** GET MPI INFO */
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  /** INITIALIZE GPU */
  wake_gpus(rank);
  slate::gpu_aware_mpi(false);
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  /** LOAD DATA */
  auto PT = load_matrix(path, m, n, comm);
  // slate::print("PT", PT);

  /** START THE TIMER (after dataset IO) */
  auto start = hrc::now();

  int t = rank == size - 1 ? m / size + m % size : m / size;
  int* t_sizes = (int*)calloc(size, sizeof(int));
  for (int i = 0; i < size; ++i)
    t_sizes[i] = i == size - 1 ? m / size + m % size : m / size;

  /** COMPUTE K */
  auto k_start = hrc::now();
  DnMat_t K;
  K.initialize(m, t, compute_kernel_matrix(PT));
  PT.releaseWorkspace();
  auto k_elapsed = get_time_elapsed(k_start);

  /** INITIALIZE V */
  auto vi_start = hrc::now();
  L_t ell;
  ell.round_robin_initialize(m, t, k, t_sizes);
  V_t V;
  V.initialize(m, t, k);
  reinit_V(V, ell);
  auto vi_elapsed = get_time_elapsed(vi_start);

  /** ALLOCATE VARIABLES */
  DnMat_t E;
  E.initialize(k, t);
  DnVec_t z, c;
  z.initialize(t);
  c.initialize(k);

  /** CREATE BENCHMARK VARIABLES */
  int64_t e_elapsed = 0;
  int64_t z_elapsed = 0;
  int64_t c_elapsed = 0;
  int64_t vr_elapsed = 0;

  /** K MEANS CLUSTERING LOOP */
  int niter = 100;
  for (int i = 0; i < niter; ++i) {
    auto e_start = hrc::now();
    spmm(handle, V, K, E);  // SpMM: ET = VK using global V
    e_elapsed += get_time_elapsed(e_start);

    auto z_start = hrc::now();
    compute_z(V, E, z);  // Calculate z from the mask of local V on ET
    z_elapsed += get_time_elapsed(z_start);

    auto c_start = hrc::now();
    spmv(handle, V, z, c);  // SpMV: c = Vz using local V
    c.sum(comm);            // Calculate global c by summing across ranks
    c_elapsed += get_time_elapsed(c_start);

    auto vr_start = hrc::now();
    ell.d_initialize(E, c);  // Calculate updated local cluster assignments
    reinit_V(V, ell);        // Reinitialize V with updated assignments
    vr_elapsed += get_time_elapsed(vr_start);
  }

  /** PRINT TIMES */
  auto elapsed = get_time_elapsed(start);
  if (rank == 0) {
    std::cout << elapsed << std::endl;
    std::cout << k_elapsed << std::endl;
    std::cout << vi_elapsed << std::endl;
    std::cout << e_elapsed << std::endl;
    std::cout << z_elapsed << std::endl;
    std::cout << c_elapsed << std::endl;
    std::cout << vr_elapsed << std::endl;
  }

  /** SAVE ASSIGNMENTS */
  std::string path_out = std::string(path) + "_out";
  ell.save(path_out.c_str(), comm);

  /** DESTROY */
  K.destroy();
  ell.destroy();
  V.destroy();
  E.destroy();
  z.destroy();
  c.destroy();
  free(t_sizes);
  cusparseDestroy(handle);

  return EXIT_SUCCESS;
}

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

  /** PARSE ARGUMENTS */
  assert(argc == 5 && "Invalid args. Must provide params [path] [m] [n] [k]");
  char* path = argv[1];
  int m = std::atoi(argv[2]);
  int n = std::atoi(argv[3]);
  int k = std::atoi(argv[4]);

  /** CLUSTER POINTS */
  cluster(path, m, n, k, comm);

  /** EXIT */
  MPI_Finalize();
  return 0;
}
