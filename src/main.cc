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
#include "cpop/utils.hh"

using namespace cpop;

using hrc = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

void destroy(cusparseSpMatDescr_t& M) {
  // Get input information
  int64_t rows, cols, nnz;
  void *cooRowInd, *cooColInd, *cooValues;
  cusparseIndexType_t idxType;
  cusparseIndexBase_t idxBase;
  cudaDataType valueType;
  cusparseCooGet(M, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues,
                 &idxType, &idxBase, &valueType);

  // Free resources
  cudaFree(cooRowInd);
  cudaFree(cooColInd);
  cudaFree(cooValues);
  cusparseDestroySpMat(M);
}

void destroy(cusparseDnMatDescr_t& M) {
  // Get input information
  void* values;
  cusparseDnMatGetValues(M, &values);

  // Free resources
  cudaFree(values);
  cusparseDestroyDnMat(M);
}

void destroy(cusparseDnVecDescr_t& V) {
  // Get input information
  void* values;
  cusparseDnVecGetValues(V, &values);

  // Free resources
  cudaFree(values);
  cusparseDestroyDnVec(V);
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
  auto k_elapsed = get_time_elapsed(k_start);

  /** INITIALIZE V */
  auto vi_start = hrc::now();
  auto V = initialize_v(handle, m, k, comm);
  auto vi_elapsed = get_time_elapsed(vi_start);

  /** K MEANS CLUSTERING LOOP */
  int niter = 1;
  for (int i = 0; i < niter; ++i) {
    /** SPMM ET = VK */
    auto ET = spmm(handle, V, K_loc);

    /** SPMV c = Vz */
    auto c = compute_c(handle, V, ET, comm);

    destroy(V);
    destroy(ET);
    destroy(c);
  }

  /** PRINT TIMES */
  if (rank == 0) {
    std::cout << "Time K: " << k_elapsed << "ms" << std::endl;
    std::cout << "Time Init V: " << vi_elapsed << "ms" << std::endl;
  }

  /** DESTROY */
  destroy(K_loc);
  cusparseDestroy(handle);

  /** EXIT */
  MPI_Finalize();
  return 0;
}
