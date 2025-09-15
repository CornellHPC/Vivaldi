#pragma once
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <mpi.h>
#include <cusparse_v2.h>

#include "utils.hh"


namespace cpop {

struct DistV2D {
  /**
   * Constructor that loads the specified data.
   *
   * Each rank should supply a specific range of rows. In
   * particular, for a sparse matrix with m rows distributed
   * amongst p processors, rank i is responsible for rows in
   * the range [i*m//p, (i+1)*m//p). Each point is formed by
   * one entry in rows, cols, and vals.
   *
   * @param row_ids is a vector of row indices for each entry
   * @param col_ids is a vector of column indices for each entry
   * @param vals is a vector of values for each entry
   * @param rows is num of rows
   * @param cols is num of cols
   * @param comm is the MPI communicator used for the matrix distribution
   */
  DistV2D(std::vector<int>& row_ids, std::vector<int>& col_ids,
            std::vector<float>& vals, int64_t rows, int64_t cols,
            MPI_Comm comm, Handle& handle);


  int map2d(float i, float j);

  /**
   * Initializes and returns the V matrix for popcorn.
   * This does a round-robin assignment of points to clusters.
   *
   * @param points is the number of points to cluster
   * @param k is the number of clusters
   * @param comm is the MPI communicator
   */
  static DistV2D initialize_v(Handle& handle, int64_t points, int64_t k, MPI_Comm comm);



  /**
   * Saves the cluster assignments to disk.
   * It computes the final assignment based on the distance matrix.
   *
   * @param filename is the name of the output file
   */
  void save_assignments(const char* filename);

  /**
   * Transposes the sparse matrix in-place.
   */
  void transpose();

  /**
   * Converts COO format vectors to cuSPARSE CSR format stored on device.
   * Takes the COO vectors from the constructor and converts them to CSR format.
   *
   * @param coo_row vector of row indices in COO format
   * @param coo_col vector of column indices in COO format
   * @param coo_val vector of values in COO format
   * @param nnz number of non-zero elements
   */
  void convertCOOtoCSR(const std::vector<int>& coo_row,
                       const std::vector<int>& coo_col,
                       const std::vector<float>& coo_val,
                       int64_t nnz, Handle& handle);


  // Number of rows in the global matrix
  int64_t rows, local_rows;

  // Number of columns in the global matrix
  int64_t cols, local_cols;

  // Number of nonzeros in the global matrix
  int64_t nonzeros, local_nnz;

  // Grid total size
  int grid_dim, world_size;

  MPI_Comm row_world, col_world, world;
  int row_rank, col_rank, world_rank;

  float * d_values;
  int * d_colinds, * d_rowptrs;

  // cuSPARSE sparse matrix descriptor
  cusparseSpMatDescr_t csr_mat;

  // Destructor to clean up cuSPARSE resources
  ~DistV2D();

};
} 
