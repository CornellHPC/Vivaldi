// C++ standard imports
#include <algorithm>
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

SparseMat SparseMat::initialize_v(ClusterAssignment &assignment,
                                  MPI_Comm comm) {
  // Get global assignment data
  int m = assignment.get_total_points();
  int k = assignment.get_total_clusters();
  std::vector<float> points = assignment.get_points();
  std::vector<float> clusters = assignment.get_clusters();
  std::vector<float> points_per_cluster = assignment.get_points_per_cluster();

  // Get communicator information
  int rank, grid_dim, P;
  MPI_Comm_rank(comm, &rank);
  grid_dim = square_grid_dim(comm);
  P = grid_dim * grid_dim;

  // Find local range of points
  int start = (m / P) * rank + std::min(m % P, rank);
  int end = start + (m / P) + ((rank < m % P) ? 1 : 0);

  // Compute local entries
  std::vector<float> lrows, lcols;
  std::vector<DATA_TYPE> lvals;
  for (int point = start; point < end; ++point) {
    int cluster = clusters.at(point);
    lrows.push_back(cluster);
    lcols.push_back(point);
    lvals.push_back(((DATA_TYPE)1) / points_per_cluster.at(cluster));
  }

  // Initialize CombBLAS communicator grid
  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(comm, grid_dim, grid_dim);

  return SparseMat(lrows, lcols, lvals, k, m, comm);
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

  return DenseMat(std::move(O), R.comm);
}

} // namespace popcorn
