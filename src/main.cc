#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include "mpi.h"

#include "cpop/cluster.hh"
#include "cpop/compute_kernel.hh"
#include "cpop/utils.hh"
#include "cpop/dist_v.hh"

using namespace cpop;

using hrc = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

int cluster_15d(ArgParse args, MPI_Comm comm) 
{
    //TODO
}

int cluster2d(ArgParse args, MPI_Comm comm) 
{
  Timer timer;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  wake_gpus(rank);
  slate::gpu_aware_mpi(true);

  /** Load Data (IO) */
#ifndef BASIC
  auto io_start = hrc::now();
#endif

  auto PT = load_matrix2d(args.path.c_str(), args.m, args.n, comm);

#ifndef BASIC
  MPI_Barrier(comm);
  timer.io = get_time_elapsed(io_start);
#endif

  /** IO done, start the main timer! */
  auto start = hrc::now();

  /** Initialize handle */
  Handle handle(args.s);

  /** Initialize V matrix */
#ifndef BASIC
  auto vi_start = hrc::now();
#endif

  auto V = DistV2D::initialize_v(handle, args.m, args.k, comm);

#ifndef BASIC
  MPI_Barrier(comm);
  timer.vi_elapsed = get_time_elapsed(vi_start);
#endif

/** Initialize K matrix */
#ifndef BASIC
  auto k_start = hrc::now();
#endif

  DnMat_t K(V.local_cols, V.local_cols, compute_kernel_matrix2d(handle, PT, args.gamma, args.c, args.r, false));
  PT.releaseWorkspace();

#ifndef BASIC
  MPI_Barrier(comm);
  timer.k_elapsed = get_time_elapsed(k_start);
#endif

  /** Initialize E, z, and c */
  DnMat_t E(V.local_rows, V.local_cols);
  DnVec_t z(V.local_rows);
  DnVec_t c(args.k);

#ifndef BASIC
  timer.dead_proc_counts =
      (int*)calloc(args.niter, sizeof(int));  // initialize dead process counts
#endif

  /** K-Means Loop */
  for (int i = 0; i < args.niter; ++i) {
    timer.niter += 1;  // Increment iteration counter

#ifndef BASIC
    auto e_start = hrc::now();
#endif


    spmm2d(handle, V, K, E);  // SpMM: ET = VK using global V


#ifndef BASIC
    MPI_Barrier(comm);
    timer.e_elapsed += get_time_elapsed(e_start);
    auto z_start = hrc::now();
#endif

    compute_z2d(V, E, z);  // Calculate z from the mask of local V on ET


#ifndef BASIC
    MPI_Barrier(comm);
    timer.z_elapsed += get_time_elapsed(z_start);
    auto c_start = hrc::now();
    auto c_computation_start = hrc::now();
#endif

    //spmv(handle, V, z, c);  // SpMV: c = Vz using local V
                            //
#ifndef BASIC
    MPI_Barrier(comm);
    timer.c_computation += get_time_elapsed(c_computation_start);
    auto c_mpi_start = hrc::now();
#endif

    sum_vec2d(c, comm);  // Calculate global c by summing across ranks
                       //
#ifndef BASIC
    MPI_Barrier(comm);
    timer.c_mpi += get_time_elapsed(c_mpi_start);
    timer.c_elapsed += get_time_elapsed(c_start);
    auto vr_start = hrc::now();
    auto vr_computation_start = hrc::now();
#endif

    argmin2d(E, c, V);  // Argmin kernel (compute D matrix)


#ifndef BASIC
    vr_computation_start = hrc::now();
#endif

    set_V_from_assignments2d(E, c, V);  // Reinitialize V based on D matrix
                                      
#ifndef BASIC
    MPI_Barrier(comm);
    timer.vr_computation += get_time_elapsed(vr_computation_start);
    timer.vr_elapsed += get_time_elapsed(vr_start);
#endif

  }

  /** Save and exit */
  MPI_Barrier(comm);
  timer.elapsed = get_time_elapsed(start);
  //timer.save_all(args.benchmark.c_str(), compute_cluster_score(K, E, c, V));
  //V.save(args.output.c_str());
  return EXIT_SUCCESS;
}

/**
 * @brief Cluster the data using the popcorn kernel k-means algorithm.
 * 
 * @param args argument parser
 * @param comm MPI communicator
 * @return EXIT_SUCCESS on success
 */
int cluster1d(ArgParse args, MPI_Comm comm) 
{
  Timer timer;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  wake_gpus(rank);
  slate::gpu_aware_mpi(true);

  /** Load Data (IO) */
#ifndef BASIC
  auto io_start = hrc::now();
#endif

#ifdef GEMM_2D
  auto PT = load_matrix2d(args.path.c_str(), args.m, args.n, comm);
#else
  auto PT = load_matrix(args.path.c_str(), args.m, args.n, comm);
#endif

#ifndef BASIC
  MPI_Barrier(comm);
  timer.io = get_time_elapsed(io_start);
#endif

  /** IO done, start the main timer! */
  auto start = hrc::now();

  /** Initialize handle */
  Handle handle(args.s);

  /** Initialize V matrix */
#ifndef BASIC
  auto vi_start = hrc::now();
#endif
  V_t V(args.m, args.k, args.s, comm);
  int64_t t = V.t;  // get this process tile size
#ifndef BASIC
  MPI_Barrier(comm);
  timer.vi_elapsed = get_time_elapsed(vi_start);
#endif

/** Initialize K matrix */
#ifndef BASIC
  auto k_start = hrc::now();
#endif

#ifdef GEMM_2D
  DnMat_t K(args.m, t, compute_kernel_matrix2d(handle, PT, args.gamma, args.c, args.r, true));
#else
  DnMat_t K(args.m, t, compute_kernel_matrix(PT, args.gamma, args.c, args.r));
#endif
  PT.releaseWorkspace();

#ifndef BASIC
  MPI_Barrier(comm);
  timer.k_elapsed = get_time_elapsed(k_start);
#endif

  /** Initialize E, z, and c */
  DnMat_t E(args.k, t);
  DnVec_t z(t);
  DnVec_t c(args.k);

#ifndef BASIC
  timer.dead_proc_counts =
      (int*)calloc(args.niter, sizeof(int));  // initialize dead process counts
#endif

  /** K-Means Loop */
  for (int i = 0; i < args.niter; ++i) {
    timer.niter += 1;  // Increment iteration counter
#ifndef BASIC
    auto e_start = hrc::now();
#endif
    spmm(handle, V, K, E);  // SpMM: ET = VK using global V
#ifndef BASIC
    MPI_Barrier(comm);
    timer.e_elapsed += get_time_elapsed(e_start);
    auto z_start = hrc::now();
#endif

    compute_z(V, E, z);  // Calculate z from the mask of local V on ET
#ifndef BASIC
    MPI_Barrier(comm);
    timer.z_elapsed += get_time_elapsed(z_start);
    auto c_start = hrc::now();
    auto c_computation_start = hrc::now();
#endif

    spmv(handle, V, z, c);  // SpMV: c = Vz using local V
                            //
#ifndef BASIC
    MPI_Barrier(comm);
    timer.c_computation += get_time_elapsed(c_computation_start);
    auto c_mpi_start = hrc::now();
#endif

    sum_vec(c, comm);  // Calculate global c by summing across ranks
                       //
#ifndef BASIC
    MPI_Barrier(comm);
    timer.c_mpi += get_time_elapsed(c_mpi_start);
    timer.c_elapsed += get_time_elapsed(c_start);
    auto vr_start = hrc::now();
    auto vr_computation_start = hrc::now();
#endif

    argmin(E, c, V);  // Argmin kernel (compute D matrix)

#ifndef BASIC
    MPI_Barrier(comm);
    timer.vr_computation += get_time_elapsed(vr_computation_start);
    auto vr_mpi_start = hrc::now();
#endif
    // Gather assignments and clusters
    int dead_process_count = gather_assignments(E, c, V, args.convergence);
    // Record dead process count at iteration
    timer.dead_proc_counts[i] = dead_process_count;
    bool done = (dead_process_count == V.n_procs);

#ifndef BASIC
    MPI_Barrier(comm);
    timer.vr_mpi += get_time_elapsed(vr_mpi_start);
#endif

    if (args.convergence && done)
      break;  // All processes are dead, so quit

#ifndef BASIC
    vr_computation_start = hrc::now();
#endif

    set_V_from_assignments(E, c, V);  // Reinitialize V based on D matrix
                                      
#ifndef BASIC
    MPI_Barrier(comm);
    timer.vr_computation += get_time_elapsed(vr_computation_start);
    timer.vr_elapsed += get_time_elapsed(vr_start);
#endif

  }

  /** Save and exit */
  MPI_Barrier(comm);
  timer.elapsed = get_time_elapsed(start);
  timer.save_all(args.benchmark.c_str(), compute_cluster_score(K, E, c, V));
  V.save(args.output.c_str());
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
  cluster1d(args, comm);

  /** Exit */
  MPI_Finalize();
  return 0;
}
