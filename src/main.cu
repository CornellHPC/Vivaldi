#include <cassert>
#include <cmath>
#include <cstdlib>

#include "mpi.h"

#include "cpop/cluster.hh"
#include "cpop/compute_kernel.hh"
#include "cpop/utils.hh"
#include "cpop/dist_v.hh"

using namespace cpop;

/****************
 * 2D Clustering
 * **************/

int cluster2d(ArgParse args, MPI_Comm comm) 
{
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

  /** Initialize process grid */
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int grid_size = std::floor(std::sqrt(world_size));
  std::shared_ptr<ProcessGrid> grid = std::make_shared<ProcessGrid>(grid_size, grid_size, true);

  /** Initialize V matrix */
#ifndef BASIC
  auto vi_start = hrc::now();
#endif

#ifdef DEBUG2D
  print_phase("Making V");
#endif


  DistV2D V(args.m, args.k, grid);


#ifndef BASIC
  MPI_Barrier(comm);
  timer.vi_elapsed = get_time_elapsed(vi_start);
#endif

/** Initialize K matrix */
#ifndef BASIC
  auto k_start = hrc::now();
#endif

#ifdef DEBUG2D
  print_phase("Making K");
#endif

  DistDnMat_t K({new DnMat_t(V.cols, V.cols, compute_kernel_matrix2d(handle, PT, args.gamma, args.c, args.r, false)),
                 grid});
  PT.releaseWorkspace();


#ifndef BASIC
  MPI_Barrier(comm);
  timer.k_elapsed = get_time_elapsed(k_start);
#endif


  /** Initialize E, z, and c */
  DistDnMat_t E({new DnMat_t(V.rows, V.cols), 
                 grid});
  DistDnVec_t z({new DnVec_t(V.cols), grid});
  DistDnVec_t c({new DnVec_t(V.rows), grid});


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


#ifdef DEBUG2D
    print_phase("SpMM");
#endif

    spmm2d(handle, V, K, E);  


#ifndef BASIC
    MPI_Barrier(comm);
    timer.e_elapsed += get_time_elapsed(e_start);
    auto z_start = hrc::now();
#endif


#ifdef DEBUG2D
    print_phase("Z");
#endif

    compute_z2d(V, E, z);  // Calculate z from the mask of local V on ET


#ifndef BASIC
    MPI_Barrier(comm);
    timer.z_elapsed += get_time_elapsed(z_start);
    auto c_start = hrc::now();
    auto c_computation_start = hrc::now();
#endif

#ifdef DEBUG2D
    print_phase("SpMV");
#endif

    spmv(handle, V, *z.vec, *c.vec);  // SpMV: c = Vz using local V

#ifndef BASIC
    MPI_Barrier(comm);
    timer.c_computation += get_time_elapsed(c_computation_start);
    auto c_mpi_start = hrc::now();
#endif

#ifdef DEBUG2D
    print_phase("Sum");
#endif

    sum_vec2d(c);  // Calculate global c by summing across ranks

#ifndef BASIC
    MPI_Barrier(comm);
    timer.c_mpi += get_time_elapsed(c_mpi_start);
    timer.c_elapsed += get_time_elapsed(c_start);
    auto vr_start = hrc::now();
    auto vr_computation_start = hrc::now();
#endif


#ifdef DEBUG2D
    print_phase("Argmin");
#endif

    argmin2d(E, c, V);  // Argmin kernel (compute D matrix)


#ifndef BASIC
    vr_computation_start = hrc::now();
#endif

#ifdef DEBUG2D
    print_phase("Reinit");
#endif

    set_V_from_assignments2d(V);  // Reinitialize V based on D matrix
                                      
#ifndef BASIC
    MPI_Barrier(comm);
    timer.vr_computation += get_time_elapsed(vr_computation_start);
    timer.vr_elapsed += get_time_elapsed(vr_start);
#endif

  }

  if (rank==0)
  {
      std::cout<<"Done 2D clustering"<<std::endl;
  }

  /** Save and exit */
  MPI_Barrier(comm);
  timer.elapsed = get_time_elapsed(start);
  float score = 0.0f; //TODO -- 2d cluster score
  timer.save_all(args.benchmark.c_str(), score);
  return EXIT_SUCCESS;
}



/****************
 * 1.5D Clustering
 * **************/

int cluster15d(ArgParse args, MPI_Comm comm) 
{
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

  /** Initialize process grid */
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int grid_size = std::floor(std::sqrt(world_size));
  std::shared_ptr<ProcessGrid> grid2d = std::make_shared<ProcessGrid>(grid_size, grid_size, true);
  std::shared_ptr<ProcessGrid> grid1d = std::make_shared<ProcessGrid>(world_size, 1, false);

  /** Initialize V matrix */
#ifndef BASIC
  auto vi_start = hrc::now();
#endif


  DistV1D V(args.m, args.k, args.s, grid1d);


#ifndef BASIC
  MPI_Barrier(comm);
  timer.vi_elapsed = get_time_elapsed(vi_start);
#endif

/** Initialize K matrix */
#ifndef BASIC
  auto k_start = hrc::now();
#endif


  DistDnMat_t K({new DnMat_t(V.local_v->t*grid_size, V.local_v->t*grid_size, 
                             compute_kernel_matrix2d(handle, 
                                                     PT, 
                                                     args.gamma, 
                                                     args.c, 
                                                     args.r, 
                                                     false)),
                             grid2d});
  PT.releaseWorkspace();


#ifndef BASIC
  MPI_Barrier(comm);
  timer.k_elapsed = get_time_elapsed(k_start);
#endif


  /** Initialize E, z, and c */
  int64_t row_tile_size = tile_dim(grid2d->col_comm, V.local_v->k_);
  DistDnMat_t E({new DnMat_t(V.local_v->k_, V.local_v->t), 
                 grid1d});
  DistDnMat_t E_p({new DnMat_t(V.local_v->k_, K.mat->w_), 
                  grid2d});
  DistDnVec_t z({new DnVec_t(V.local_v->t), grid1d});
  DistDnVec_t c({new DnVec_t(V.local_v->k_), grid1d});


  /** Temporary buffer */
  float * d_tmp;
  //TODO: This actually overallocates a bit
  CHECK_CUDA(cudaMalloc(&d_tmp, sizeof(float) * V.local_v->k_ * E_p.mat->w_));

  float * d_tmp2;
  CHECK_CUDA(cudaMalloc(&d_tmp2, sizeof(float) * V.local_v->k_ * E.mat->w_));

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


    spmm15d(handle, V, K, E, E_p, d_tmp, d_tmp2);  


#ifndef BASIC
    MPI_Barrier(comm);
    timer.e_elapsed += get_time_elapsed(e_start);
    auto z_start = hrc::now();
#endif


    compute_z(*V.local_v, *E.mat, *z.vec);  // Calculate z from the mask of local V on ET


#ifndef BASIC
    MPI_Barrier(comm);
    timer.z_elapsed += get_time_elapsed(z_start);
    auto c_start = hrc::now();
    auto c_computation_start = hrc::now();
#endif

    spmv(handle, *V.local_v, *z.vec, *c.vec);  // SpMV: c = Vz using local V

#ifndef BASIC
    MPI_Barrier(comm);
    timer.c_computation += get_time_elapsed(c_computation_start);
    auto c_mpi_start = hrc::now();
#endif

    sum_vec(*c.vec, grid1d->world_comm);  // Calculate global c by summing across ranks

#ifndef BASIC
    MPI_Barrier(comm);
    timer.c_mpi += get_time_elapsed(c_mpi_start);
    timer.c_elapsed += get_time_elapsed(c_start);
    auto vr_start = hrc::now();
    auto vr_computation_start = hrc::now();
#endif

    argmin(*E.mat, *c.vec, *V.local_v, true);  // Argmin kernel (compute D matrix)


#ifndef BASIC
    vr_computation_start = hrc::now();
#endif

    set_V_from_assignments15d(V);  // Reinitialize V based on D matrix
                                      
#ifndef BASIC
    MPI_Barrier(comm);
    timer.vr_computation += get_time_elapsed(vr_computation_start);
    timer.vr_elapsed += get_time_elapsed(vr_start);
#endif

  }

  CHECK_CUDA(cudaFree(d_tmp));
  CHECK_CUDA(cudaFree(d_tmp2));

  if (rank==0)
  {
      std::cout<<"Done 1.5D clustering"<<std::endl;
  }

  /** Save and exit */
  MPI_Barrier(comm);
  timer.elapsed = get_time_elapsed(start);
  timer.save_all(args.benchmark.c_str(), 0.0); //TODO: cluster score computation
  V.local_v->save(args.output.c_str());
  return EXIT_SUCCESS;
}



/****************
 * 1D Clustering
 * **************/
/**
 * @brief Cluster the data using the popcorn kernel k-means algorithm.
 * 
 * @param args argument parser
 * @param comm MPI communicator
 * @return EXIT_SUCCESS on success
 */
int cluster1d(ArgParse args, MPI_Comm comm) 
{
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
    CHECK_CUDA(cudaDeviceSynchronize());

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
    CHECK_CUDA(cudaDeviceSynchronize());

#ifndef BASIC
    MPI_Barrier(comm);
    timer.c_computation += get_time_elapsed(c_computation_start);
    auto c_mpi_start = hrc::now();
#endif

    sum_vec(c, comm);  // Calculate global c by summing across ranks
    CHECK_CUDA(cudaDeviceSynchronize());
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
    auto e_mpi_start = hrc::now();
#endif
    // Gather assignments and clusters
    int dead_process_count = gather_assignments(E, c, V, args.convergence);
    // Record dead process count at iteration
    timer.dead_proc_counts[i] = dead_process_count;
    //bool done = (dead_process_count == V.n_procs);

#ifndef BASIC
    MPI_Barrier(comm);
    timer.e_mpi += get_time_elapsed(e_mpi_start);
#endif

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

  if (rank==0)
  {
      std::cout<<"Done 1D clustering"<<std::endl;
  }

  /** Save and exit */
  MPI_Barrier(comm);
  timer.elapsed = get_time_elapsed(start);
  float score = compute_cluster_score(K, E, c, V);
  timer.save_all(args.benchmark.c_str(), score);
  if (rank==0)
  {
      std::cout<<"K-means score: "<<score<<std::endl;
  }
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

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank==0)
  {
      std::cout<<"Running clusterpop "<<args.alg<<" mode."<<std::endl;
  }

  /** Cluster */
  if (args.alg.compare("1d")==0) {
      cluster1d(args, comm);
  } else if (args.alg.compare("15d")==0) {
      cluster15d(args, comm);
  } else if (args.alg.compare("2d")==0) {
      cluster2d(args, comm);
  }

  /** Exit */
  MPI_Finalize();
  return 0;
}
