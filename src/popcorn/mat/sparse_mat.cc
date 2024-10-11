// C++ standard imports
#include <cassert>
#include <memory>

// Local imports
#include "../utils/utils.hh"
#include "dense_mat.hh"
#include "sparse_mat.hh"

namespace popcorn {

SparseMat::SparseMat(std::vector<float> &row_ids, std::vector<float> &col_ids,
                     std::vector<DATA_TYPE> &vals, int64_t rows, int64_t cols,
                     MPI_Comm comm) {
  this->grid_dim = square_grid_dim(comm);
  this->comm = comm;

  // Initialize CombBLAS communicator grid
  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(comm, grid_dim, grid_dim);

  // Initialize distributed data vectors
  combblas::FullyDistVec<int64_t, DATA_TYPE> drows(row_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dcols(col_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dvals(vals, grid);

  // Initialize distributed sparse matrix
  cm = std::make_unique<combblas::SpParMat<int64_t, DATA_TYPE, UDER>>(
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
  comm = cm->getcommgrid()->GetWorld();
  grid_dim = square_grid_dim(comm);
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
  int *points_per_cluster = (int *)calloc(sizeof(int), k);
  for (int i = 0; i < k; ++i) {
    points_per_cluster[i] = (points / k) + ((i < points % k) ? 1 : 0);
  }

  // Compute points per process assuming round-robin assignment
  int *points_per_process = (int *)calloc(sizeof(int), P);
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

void SparseMat::transpose() { cm->Transpose(); }

void SparseMat::print(std::string prefix, std::ostream &out) {
  cm->PrintInfo();
}

DenseMat SparseMat::spmm(DenseMat &R) {
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

  return DenseMat(std::move(O));
}

} // namespace popcorn
