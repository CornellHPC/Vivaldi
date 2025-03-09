#ifndef CPOP_COMPUTE_C_HH
#define CPOP_COMPUTE_C_HH

#include "cusparse.h"
#include "mpi.h"

#include "utils.hh"

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
   */
  DnMat_t(int64_t h, int64_t w);

  /**
   * @brief Constructor
   * 
   * @param h The height
   * @param w The width
   * @param dM_ The on-device array of size h * w
   */
  DnMat_t(int64_t h, int64_t w, float* dM_);

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
   */
  DnVec_t(int t);

  int print();

  ~DnVec_t();
};

struct V_t {
  bool sparse;
  cusparseSpMatDescr_t gV, lV;

  // GPU pointers to the components of the global CSC V matrix
  int64_t* global_assignments;      // CSC row indices
  int64_t* global_csc_col_offsets;  // CSC column offsets
  float* values;                    // CSC values of global V

  // GPU pointers to the components of the local CSC V matrix
  // Aside from local_csc_col_offsets, these are the same as the global ones
  //    plus some offset (which is determined by displs[rank])
  int64_t* local_ptr_to_assignments;
  int64_t* local_csc_col_offsets;
  float* local_ptr_to_values;

  // Working vectors used in V reinitialization
  int* global_cluster_sizes;
  int64_t* local_assignments;
  int* local_cluster_sizes;

  // Other basic vars
  int64_t m_, t_, k_;
  int* t_sizes_;
  int rank, n_procs;
  int* displs;
  MPI_Comm comm;

  // Host vector used in convergence checking
  int64_t* previous_global_assignments;

  /**
   * @brief Constructor
   * 
   * @param m The global number of points
   * @param t The local number of points
   * @param k The number of clusters
   * @param t_sizes n_procs-size array of tile widths for each process
   * @param sparse Flag indicating whether or not V is sparse
   * @param comm The MPI communicator used to distribute assignments
   */
  V_t(int64_t m, int64_t t, int64_t k, int* t_sizes, bool sparse,
      MPI_Comm comm);

  /**
   * @brief Saves the cluster assignments to disk
   *
   * @param path The path to the output file
   */
  int save(const char* path);

  /**
   * @brief Checks if the cluster assignments have converged on CPU
   * 
   * @return true if converged
   * @return false if not converged
   */
  bool test_convergence();

  /**
   * @brief Prints the CSC vectors for V
   * 
   * Used in debugging only
   */
  void print();

  ~V_t();
};

/**
 * @brief Computes E by SpMM routine
 * 
 * @param handle The handle
 * @param V The V matrix
 * @param K The K matrix
 * @param E The E matrix
 * @return int 
 */
int spmm(Handle& handle, V_t& V, DnMat_t& K, DnMat_t& E);

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
 * @brief Computes the local c norm vector by SpMV. Used in ``compute_c``.
 *
 * @param handle The handle
 * @param V The V matrix
 * @param z The local z vector
 * @param c The local c vector
 * @return int
 */
int spmv(Handle& handle, V_t& V, DnVec_t& z, DnVec_t& c);

/**
 * @brief Sums the vector across the communicator. Used in ``compute_c``.
 *
 * @param c The vector to sum
 * @param comm The communicator over which to sum
 * @return int
 */
int sum_vec(DnVec_t& c, MPI_Comm comm);

/**
 * @brief Computes the c norm vector by SpMV and sums it across the communicator row
 * 
 * @param handle The handle
 * @param V The V matrix
 * @param z The local z vector
 * @param c The local c vector
 * @param comm The communicator over which to sum
 * @return int 
 */
int compute_c(Handle& handle, V_t& V, DnVec_t& z, DnVec_t& c, MPI_Comm comm);

/**
 * @brief Launches the argmin kernel. Used in ``reinit_V``.
 * 
 * @param E The E matrix
 * @param c The c norm vector
 * @param V The V matrix
 * @return int
 */
int argmin(DnMat_t& E, DnVec_t& c, V_t& V);

/**
 * @brief Gathers assignments and clusters. Used in ``reinit_V``.
 * 
 * @param E The E matrix
 * @param c The c norm vector
 * @param V The V matrix
 * @return int
 */
int gather_assignments(DnMat_t& E, DnVec_t& c, V_t& V);

/**
 * @brief Launches the reinit kernel. Used in ``reinit_V``.
 * 
 * @param E The E matrix
 * @param c The c norm vector
 * @param V The V matrix
 * @return int
 */
int set_V_from_assignments(DnMat_t& E, DnVec_t& c, V_t& V);

/**
 * @brief Reinitializes V based on the distances matrix (computed from E and c)
 *
 * @param E The E matrix
 * @param c The c norm vector
 * @param V The V matrix
 * @return int
 */
int reinit_V(DnMat_t& E, DnVec_t& c, V_t& V);

}  // namespace cpop

#endif  // CPOP_COMPUTE_C_HH
