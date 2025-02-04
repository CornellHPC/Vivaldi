#ifndef CPOP_COMPUTE_C_HH
#define CPOP_COMPUTE_C_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

struct DnMat_t {
  float* dM;
  cusparseDnMatDescr_t M;

  int h_, w_;

  /**
   * @brief Constructor
   * 
   * @param h The height
   * @param w The width
   * @return int 
   */
  int initialize(int64_t h, int64_t w);

  /**
   * @brief Constructor
   * 
   * @param h The height
   * @param w The width
   * @param dM_ The on-device array of size h * w
   * @return int 
   */
  int initialize(int64_t h, int64_t w, float* dM_);

  int print();

  ~DnMat_t();
};

struct DnVec_t {
  float* dz;
  cusparseDnVecDescr_t z;
  int64_t size;

  /**
   * @brief Constructor
   * 
   * @param t The size
   * @return int 
   */
  int initialize(int t);

  int print();

  ~DnVec_t();
};

/** Struct to hold cluster sizes (l) and point assignments (a) */
struct L_t {
  // All variables here are local (e.g. on the CPU, not GPU)

  int64_t* ga;  // global point to cluster assignments (of size m)
  int64_t* la;  // local point to cluster assignments (of size t)
  int64_t* gl;  // global cluster sizes (of size k)
  int64_t* ll;  // local cluster sizes (of size k)

  int64_t m_, t_, k_;

  int rank, n_procs;
  int* t_sizes_;

  /**
   * @brief Constructor
   * 
   * @param m The global number of points
   * @param t The local number of points
   * @param k The number of clusters
   * @param t_sizes n_procs-size array of tile widths for each process
   * @return int 
   */
  int round_robin_initialize(int64_t m, int64_t t, int64_t k, int* t_sizes);

  /**
   * @brief MPI Allreduce on local clusters (populates ga using la)
   * 
   * @return int 
   */
  int gather_clusters();

  /**
   * @brief MPI Allgatherv on local assignments (populates gl using ll)
   *
   * @return int 
   */
  int gather_assignments();

  /**
   * @brief Saves the cluster assignments to a file
   *
   * @param path The path to the file
   * @param comm The MPI communicator used to distribute assignments
   * @return int
   */
  int save(const char* path, MPI_Comm comm);

  ~L_t();
};

struct V_t {
  // Local (on the CPU) values (used in initialization and reinitialization)
  int64_t *local_csr_row_offsets, *local_csr_col_inds, *local_csc_col_offsets,
      *local_csc_row_inds, *cluster_loc_ptrs;
  float *local_gV, *local_lV;

  // Device (on the GPU) values
  int64_t *csr_row_offsets, *csr_col_inds, *csc_col_offsets, *csc_row_inds;
  float* dgV;  // global V values in CSR
  float* dlV;  // local V values in CSC

  cusparseSpMatDescr_t gV, lV;

  int64_t m_, t_, k_;

  /**
   * @brief Constructor
   * 
   * @param m The global number of points
   * @param t The local number of points
   * @param k The number of clusters
   * @return int 
   */
  int initialize(int64_t m, int64_t t, int64_t k);

  /**
   * @brief Cleans the local arrays by setting them all to 0, preparing them for reinit_V
   * 
   * @return int 
   */
  int reset_local();

  /**
   * @brief Copies the local memory buffers to the device buffers and then zeroes the local buffers
   * 
   * @return int 
   */
  int cp_local();

  ~V_t();
};

/**
 * @brief Computes E by SpMM routine
 * 
 * @param handle The cuSPARSE handle
 * @param V The V matrix
 * @param K The K matrix
 * @param E The E matrix
 * @return int 
 */
int spmm(cusparseHandle_t& handle, V_t& V, DnMat_t& K, DnMat_t& E);

/**
 * @brief Computes z based on the local V matrix and E (i.e. using the masking strategy)
 * 
 * @param V The V matrix
 * @param E The local partition of the E dense matrix
 * @param z Resulting z dense vector
 * @return int 
 */
int compute_z(V_t& V, DnMat_t& E, DnVec_t& z);

/**
 * @brief Computes the c norm vector by SpMV
 *
 * @param handle The cuSPARSE handle
 * @param V The V matrix
 * @param z The local z vector
 * @param c The local c vector
 * @return int
 */
int spmv(cusparseHandle_t& handle, V_t& V, DnVec_t& z, DnVec_t& c);

/**
 * @brief Sums the vector across the communicator
 *
 * @param c The vector to sum
 * @param comm The communicator over which to sum
 * @return int
 */
int sum_vec(DnVec_t& c, MPI_Comm comm);

/**
 * @brief Reinitializes L based on the distance matrix
 *
 * @param ell The struct containing "la" (local assignments vector) and "ll" (local cluster sizes vector)
 * @param E The E matrix
 * @param c The c norm vector
 */
int reinit_ell(L_t& ell, DnMat_t& E, DnVec_t& c);

/**
 * @brief Reinitializes V based on the local assignments and cluster sizes
 * 
 * @param V The V matrix
 * @param ell The struct containing "la" (local assignments vector) and "ll" (local cluster sizes vector)
 * @return int 
 */
int reinit_V(V_t& V, L_t& ell);

}  // namespace cpop

#endif  // CPOP_COMPUTE_C_HH
