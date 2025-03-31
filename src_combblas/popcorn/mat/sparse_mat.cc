// C++ standard imports
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

// Library imports
#include "mpio.h"

// Local imports
#include "../kernel/argmin_kernel.cuh"
#include "../utils/utils.hh"
#include "dense_mat.hh"
#include "sparse_mat.hh"

namespace popcorn {

SparseMat::SparseMat(std::vector<float>& row_ids, std::vector<float>& col_ids,
                     std::vector<DATA_TYPE>& vals, int64_t rows, int64_t cols,
                     MPI_Comm comm) {
  this->grid_dim = square_grid_dim(comm);
  this->rows = rows;
  this->cols = cols;

  // Initialize CombBLAS communicator grid
  this->grid = std::make_shared<combblas::CommGrid>(comm, grid_dim, grid_dim);

  // Initialize distributed data vectors
  combblas::FullyDistVec<int64_t, DATA_TYPE> drows(row_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dcols(col_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dvals(vals, grid);

  // Initialize distributed sparse matrix
  this->cm = std::make_unique<combblas::SpParMat<int64_t, DATA_TYPE, UDER>>(
      rows, cols, drows, dcols, dvals, false);
}

SparseMat::SparseMat(
    std::unique_ptr<combblas::SpParMat<int64_t, DATA_TYPE, UDER>> M) {
  // Save pointer to matrix
  cm = std::move(M);

  // Populate private fields
  rows = cm->getnrow();
  cols = cm->getncol();
  nonzeros = cm->getnnz();
  grid = cm->getcommgrid();
  grid_dim = square_grid_dim(grid->GetWorld());
}

SparseMat SparseMat::initialize_v(int64_t points, int64_t k, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  int grid_dim = square_grid_dim(comm);
  int P = grid_dim * grid_dim;

  // Initialize CombBLAS communicator grid
  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(comm, grid_dim, grid_dim);

  // Compute points per cluster assuming round-robin assignment
  int* points_per_cluster = (int*)calloc(sizeof(int), k);
  for (int i = 0; i < k; ++i) {
    points_per_cluster[i] = (points / k) + ((i < points % k) ? 1 : 0);
  }

  // Compute points per process assuming round-robin assignment
  int* points_per_process = (int*)calloc(sizeof(int), P);
  for (int i = 0; i < P; ++i) {
    points_per_process[i] = (points / P) + ((i < points % P) ? 1 : 0);
  }

  // Compute sparse matrix row partition for which this rank is responsible
  int row_start = 0;
  for (int i = 0; i < rank; ++i) {
    row_start += points_per_process[i];
  }
  int row_end = row_start + points_per_process[rank];

  // Compute sparse matrix entry positions and values
  std::vector<float> lrow_ids, lcol_ids, lvals;
  for (int row = row_start, col = row_start % k; row < row_end;
       ++row, col = (col + 1) % k) {
    lrow_ids.push_back(row);
    lcol_ids.push_back(col);
    lvals.push_back(1.0f / points_per_cluster[col]);
  }

  // Clean up
  free(points_per_cluster);
  free(points_per_process);

  SparseMat out = SparseMat(lrow_ids, lcol_ids, lvals, points, k, comm);
  out.transpose();
  return out;
}

SparseMat SparseMat::initialize_v(float* D) {
  // Initialize parameters
  int64_t clusters = cm->getnrow();
  int64_t points = cm->getncol();
  int64_t clusters_loc = cm->getlocalrows();
  int64_t points_loc = cm->getlocalcols();
  auto grid = cm->getcommgrid();

  // Clamp local dimension
  if (clusters_loc == 0 || points_loc == 0) {
    clusters_loc = 0;
    points_loc = 0;
  }

  // ---------------------- WARNING ----------------------
  // D is transposed so local submatrix is actually of size
  // kloc by mloc (i.e. clusters are rows and points are cols)
  // ---------------------- WARNING ----------------------

  // Get global offsets
  int64_t cluster_offset =
      grid->GetRankInProcCol() * (clusters / grid->GetGridRows());
  int64_t point_offset =
      grid->GetRankInProcRow() * (points / grid->GetGridCols());

  // Compute local argmin
  auto argmin_kernel = ArgminKernel();
  auto M = argmin_kernel.kernel(clusters_loc, points_loc, cluster_offset, D);

  // Pad to target
  int64_t points_tar = points / grid->GetGridCols();
  M = (Argmin*)realloc(M, sizeof(Argmin) * points_tar);
  for (int i = points_loc; i < points_tar; ++i) {
    M[i] = Argmin{INFINITY, 0};
  }

  // Perform column reduction
  auto gM = (Argmin*)calloc(sizeof(Argmin), points_tar);
  MPI_Reduce(M, gM, points_tar, MPI_FLOAT_INT, MPI_MINLOC, 0,
             grid->GetColWorld());

  // Prepare to construct sparse matrix
  std::vector<float> lrow_ids, lcol_ids;
  std::vector<DATA_TYPE> lvals;

  // Perform row reduction on top row
  if (grid->GetRankInProcCol() == 0) {
    auto c = (int*)calloc(clusters, sizeof(int));
    auto gc = (int*)calloc(clusters, sizeof(int));
    for (int i = 0; i < points_tar; ++i) {
      Argmin a = gM[i];
      c[a.index]++;
    }

    MPI_Allreduce(c, gc, clusters, MPI_INT, MPI_SUM, grid->GetRowWorld());
    for (int i = 0; i < points_tar; ++i) {
      Argmin a = gM[i];
      lrow_ids.push_back(a.index);
      lcol_ids.push_back(point_offset + i);
      lvals.push_back(1.0f / gc[a.index]);
    }

    free(c);
    free(gc);
  }

  SparseMat out = SparseMat(lrow_ids, lcol_ids, lvals, clusters, points,
                            cm->getcommgrid()->GetWorld());

  free(M);
  free(gM);
  return out;
}

void SparseMat::save_assignments(const char* filename) {
  // Compute global offsets
  int row_offset = cm->getcommgrid()->GetRankInProcCol() *
                   (cm->getnrow() / cm->getcommgrid()->GetGridRows());
  int col_offset = cm->getcommgrid()->GetRankInProcRow() *
                   (cm->getncol() / cm->getcommgrid()->GetGridCols());

  // Open the file
  MPI_File fh;
  MPI_File_open(cm->getcommgrid()->GetWorld(), filename,
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

  // Write the assignments
  UDER* spSeq = cm->seqptr();
  for (typename UDER::SpColIter colit = spSeq->begcol();
       colit != spSeq->endcol(); ++colit) {
    for (typename UDER::SpColIter::NzIter nzit = spSeq->begnz(colit);
         nzit != spSeq->endnz(colit); ++nzit) {
      int cluster = row_offset + nzit.rowid();
      int point = col_offset + colit.colid();

      // TODO: Optimize the writes to be sequential
      MPI_File_write_at(fh, sizeof(int) * point, &cluster, 1, MPI_INT,
                        MPI_STATUS_IGNORE);
    }
  }

  // Clean up
  MPI_File_close(&fh);
}

void SparseMat::transpose() {
  cm->Transpose();
}

void SparseMat::print(std::string prefix, std::ostream& out) {
  cm->PrintInfo();
}

DenseMat SparseMat::spmm(DenseMat& R) {
  assert(cm && "Must have a CombBLAS sparse matrix!");
  assert((R.cm || R.sm) && "Must have a SLATE or CombBLAS dense matrix!");

  // Convert dense matrix to CombBLAS if
  // it is still in SLATE representation
  if (R.cm == nullptr)
    R.to_combblas();

  // Setup for SpMM
  std::unique_ptr<combblas::DnParMat<int64_t, DATA_TYPE>> O =
      std::make_unique<combblas::DnParMat<int64_t, DATA_TYPE>>();
  combblas::spmm_stats stats;

  // Perform SpMM
  *O = combblas::SpMM_sC<SR>(*cm, *R.cm, stats);

  return DenseMat(std::move(O), R.comm);
}

}  // namespace popcorn
