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
#include "gpu_kernels.cuh"


namespace cpop {

struct ProcessGrid
{
    MPI_Comm world_comm, row_comm, col_comm;
    int world_size, row_size, col_size;
    int world_rank, row_rank, col_rank;

    ProcessGrid(int p, int q, bool colmaj);
};

struct DistV2D {

  std::shared_ptr<ProcessGrid> grid;
  float * d_vals;
  int * d_colptrs;
  int * d_rowinds;
  int * d_cluster_sizes;
  int * d_mininds;
  FloatI32 * d_minpairs;

  float * d_remote_vals;
  int * d_remote_colptrs;
  int * d_remote_rowinds;

  float * d_csr_val;
  int * d_csr_colinds;
  int * d_csr_rowptrs;

  float * d_v_dense;

  int * tile_rows;
  int * tile_cols;
  int64_t * tile_nnz;

  int64_t rows, cols, nnz;
  int64_t global_rows, global_cols, global_nnz;

  cusparseSpMatDescr_t csc_mat;

  DistV2D(int64_t m, int64_t k, std::shared_ptr<ProcessGrid> grid);
  DistV2D(int64_t m, int64_t k, int loc_k, int64_t nnz, std::shared_ptr<ProcessGrid> grid);
  void init_cusparse_csc();
  void update_nnz(int64_t nnz);
  ~DistV2D();
  int map2d(int64_t rid, int64_t cid);

};
} 
