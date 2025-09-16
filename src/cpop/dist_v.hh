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

struct ProcessGrid
{
    MPI_Comm world_comm, row_comm, col_comm;
    int world_size, row_size, col_size;
    int world_rank, row_rank, col_rank;

    ProcessGrid(int p, int q);
};

struct DistV2D {

  std::shared_ptr<ProcessGrid> grid;
  float * d_vals;
  int * d_colptrs;
  int * d_rowinds;
  int * d_cluster_sizes;

  int * tile_rows;
  int * tile_cols;
  int64_t * tile_nnz;

  int64_t rows, cols, nnz;
  int64_t global_rows, global_cols, global_nnz;

  cusparseSpMatDescr_t csc_mat;

  DistV2D(int64_t m, int64_t k, std::shared_ptr<ProcessGrid> grid);
  ~DistV2D();
  int map2d(int64_t rid, int64_t cid);

};
} 
