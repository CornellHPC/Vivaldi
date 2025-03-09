#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "mpi.h"

#include "cpop/cluster.hh"
#include "cpop/compute_kernel.hh"
#include "cpop/utils.hh"

using namespace cpop;

using hrc = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

/**
 * @brief Cluster the data using the popcorn kernel k-means algorithm.
 * This is a fastest version of the algorithm that does not need any 
 * fine-grained timing or barriers and does not support convergence testing. 
 * It is used for testing the scaling of the algorithm.
 * 
 * @param args argument parser
 * @param comm MPI communicator
 * @return EXIT_SUCCESS on success
 */
int cluster_basic(ArgParse args, MPI_Comm comm) {
  Timer timer;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  wake_gpus(rank);
  slate::gpu_aware_mpi(true);

  Handle handle(args.s);

  auto PT = load_matrix(args.path.c_str(), args.m, args.n, comm);

  int t = args.m / size + (args.m > size * size && args.m % size > 0);
  int* t_sizes = (int*)calloc(size, sizeof(int));
  for (int i = 0; i < size - 1; ++i) {
    t_sizes[i] = t;
  }
  t_sizes[size - 1] = args.m - (size - 1) * t;
  t = t_sizes[rank];

  auto start = hrc::now();
  DnMat_t K(args.m, t, compute_kernel_matrix(PT, args.gamma, args.c, args.r));
  PT.releaseWorkspace();
  V_t V(args.m, t, args.k, t_sizes, args.s, comm);
  DnMat_t E(args.k, t);
  DnVec_t z(t);
  DnVec_t c(args.k);
  for (int i = 0; i < args.niter; ++i) {
    spmm(handle, V, K, E);  // SpMM: ET = VK using global V
    compute_z(V, E, z);     // Calculate z from the mask of local V on ET
    compute_c(handle, V, z, c, comm);  // SpMV: c = Vz using local V & allreduce
    reinit_V(E, c, V);  // Reinitialize V based on D matrix (i.e. E and c)
  }
  MPI_Barrier(comm);
  timer.elapsed = get_time_elapsed(start);
  timer.save_elapsed(args.benchmark.c_str());
  V.save(args.output.c_str());
  free(t_sizes);
  return EXIT_SUCCESS;
}

/**
 * @brief Cluster the data using the popcorn kernel k-means algorithm.
 * This is the full version of the algorithm that includes timing and
 * barriers. It is used for benchmarking the algorithm.
 *
 * @param args argument parser
 * @param comm MPI communicator
 * @return EXIT_SUCCESS on success
 */
int cluster_full(ArgParse args, MPI_Comm comm) {
  /** Timing */
  Timer timer;

  /** MPI Info */
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  /** Initialize GPU */
  wake_gpus(rank);
  slate::gpu_aware_mpi(true);

  /** Create handle */
  Handle handle(args.s);

  /** Load Data (IO) */
  auto io_start = hrc::now();
  auto PT = load_matrix(args.path.c_str(), args.m, args.n, comm);

  /** IO complete, start the timer */
  MPI_Barrier(comm);
  timer.io = get_time_elapsed(io_start);
  auto start = hrc::now();

  /** Compute tile sizes */
  int t = args.m / size + (args.m > size * size && args.m % size > 0);
  int* t_sizes = (int*)calloc(size, sizeof(int));
  for (int i = 0; i < size - 1; ++i) {
    t_sizes[i] = t;
  }
  t_sizes[size - 1] = args.m - (size - 1) * t;
  t = t_sizes[rank];

  /** Compute K */
  auto k_start = hrc::now();
  DnMat_t K(args.m, t, compute_kernel_matrix(PT, args.gamma, args.c, args.r));
  PT.releaseWorkspace();
  MPI_Barrier(comm);
  timer.k_elapsed = get_time_elapsed(k_start);

  /** Initialize V */
  auto vi_start = hrc::now();
  V_t V(args.m, t, args.k, t_sizes, args.s, comm);
  timer.vi_elapsed = get_time_elapsed(vi_start);

  /** Allocations */
  DnMat_t E(args.k, t);
  DnVec_t z(t);
  DnVec_t c(args.k);

  /** K-Means Loop */
  for (int i = 0; i < args.niter; ++i) {
    auto e_start = hrc::now();
    spmm(handle, V, K, E);  // SpMM: ET = VK using global V
    timer.e_elapsed += get_time_elapsed(e_start);

    auto z_start = hrc::now();
    compute_z(V, E, z);  // Calculate z from the mask of local V on ET
    timer.z_elapsed += get_time_elapsed(z_start);

    auto c_start = hrc::now();
    auto c_computation_start = hrc::now();
    spmv(handle, V, z, c);  // SpMV: c = Vz using local V
    timer.c_computation += get_time_elapsed(c_computation_start);
    auto c_mpi_start = hrc::now();
    sum_vec(c, comm);  // Calculate global c by summing across ranks
    MPI_Barrier(comm);
    timer.c_mpi += get_time_elapsed(c_mpi_start);
    timer.c_elapsed += get_time_elapsed(c_start);

    auto vr_start = hrc::now();
    auto vr_computation_start = hrc::now();
    argmin(E, c, V);  // Argmin kernel (compute D matrix)
    timer.vr_computation += get_time_elapsed(vr_computation_start);
    auto vr_mpi_start = hrc::now();
    gather_assignments(E, c, V);  // Gather assignments and clusters
    MPI_Barrier(comm);
    timer.vr_mpi += get_time_elapsed(vr_mpi_start);
    vr_computation_start = hrc::now();
    set_V_from_assignments(E, c, V);  // Reinitialize V based on D matrix
    timer.vr_computation += get_time_elapsed(vr_computation_start);
    timer.vr_elapsed += get_time_elapsed(vr_start);

    if (args.convergence) {
      // Rank 1 tests convergence on CPU
      bool converged = false;
      if (rank == 0 && V.test_convergence()) {
        converged = true;
        std::cout << "Converged at iteration " << i << std::endl;
      }
      // Broadcast convergence to all ranks
      MPI_Bcast(&converged, 1, MPI_C_BOOL, 0, comm);
      if (converged)
        break;  // All ranks exit the loop when converged
    }
  }

  /** Save benchmarking */
  MPI_Barrier(comm);
  timer.elapsed = get_time_elapsed(start);
  timer.save_all(args.benchmark.c_str());

  /** Save assignments */
  V.save(args.output.c_str());

  /** Exit */
  free(t_sizes);
  return EXIT_SUCCESS;
}

/**
 * Runs the distributed popcorn kernel k-means clustering algorithm.
 * For help, run ``srun cpop --help``
 */
int main(int argc, char* argv[]) {
  /** Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  /** Argument Parsing */
  ArgParse args(argc, argv);

  /** Cluster */
  if (args.basic)
    cluster_basic(args, comm);
  else
    cluster_full(args, comm);

  /** Exit */
  MPI_Finalize();
  return 0;
}
