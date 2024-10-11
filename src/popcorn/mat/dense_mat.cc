// C++ standard imports
#include <cassert>
#include <memory>

// Local imports
#include "../utils/utils.hh"
#include "dense_mat.hh"
#include "sparse_mat.hh"

namespace popcorn {

DenseMat DenseMat::load_from_file(const char *filename, int64_t rows,
                                  int64_t cols, MPI_Comm comm) {

  // Open file
  MPI_File fh;
  MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Read data
  size_t count = rows * cols * sizeof(DATA_TYPE);
  DATA_TYPE *data = (DATA_TYPE *)malloc(count);
  MPI_File_read_all(fh, data, rows * cols, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Compute tile size
  int grid_dim = square_grid_dim(comm);
  int rows_per_block = tile_dim(comm, rows);
  int cols_per_block = tile_dim(comm, cols);

  // Create empty SLATE matrix object of size equal to data
  auto M = std::make_unique<slate::Matrix<DATA_TYPE>>(
      rows, cols, rows_per_block, cols_per_block, grid_dim, grid_dim, comm);
  M->insertLocalTiles(slate::Target::Devices);

  // Fill data
  int64_t m = M->m(), n = M->n(), mb = M->tileMb(0), nb = M->tileNb(0);
  for (int64_t j = 0; j < M->nt(); ++j) {   // i loops over block columns
    for (int64_t i = 0; i < M->mt(); ++i) { // j loops over block rows
      if (M->tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = M->at(i, j, M->tileDevice(i, j));
        int64_t lda = tile.stride();
        DATA_TYPE *A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) { // jj loops over columns
          int64_t global_column = j * nb + jj;
          int64_t global_row_start = i * mb;
          cudaMemcpy(A + jj * lda, data + global_column * m + global_row_start,
                     sizeof(DATA_TYPE) * tile.mb(), cudaMemcpyHostToDevice);
        }
      }
    }
  }

  // Clean up
  free(data);
  MPI_File_close(&fh);

  return DenseMat(std::move(M));
}

DenseMat DenseMat::transpose() {
  assert(sm && "Can only transpose SLATE matrices!");

  std::unique_ptr<slate::Matrix<DATA_TYPE>> M =
      std::make_unique<slate::Matrix<DATA_TYPE>>();
  *M = slate::transpose(*sm);

  return DenseMat(std::move(M));
}

DenseMat::DenseMat(std::unique_ptr<slate::Matrix<DATA_TYPE>> M) {
  // Save pointer to matrix
  sm = std::move(M);

  // Populate private fields
  rows = sm->m();
  cols = sm->n();
  block_rows = sm->mt();
  block_cols = sm->nt();
  rows_per_block = sm->tileMb(0);
  cols_per_block = sm->tileNb(0);
  grid_dim = square_grid_dim(sm->mpiComm());
  comm = sm->mpiComm();
}

DenseMat::DenseMat(std::unique_ptr<combblas::DnParMat<int64_t, DATA_TYPE>> M) {
  // Save pointer to matrix
  cm = std::move(M);

  // Populate private fields
  rows = cm->getgnrow();
  cols = cm->getgncol();
  // TODO: Populate the rest of the stuff
}

void DenseMat::to_combblas() {
  assert(sm && "SLATE matrix must exist!");

  // Initialize CombBLAS communicator grid
  MPI_Comm comm = sm->mpiComm();
  int p = square_grid_dim(comm);
  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(comm, p, p);

  // Initialize CombBLAS distributed dense matrix
  std::unique_ptr<combblas::DnParMat<int64_t, DATA_TYPE>> C =
      std::make_unique<combblas::DnParMat<int64_t, DATA_TYPE>>(
          grid, sm->m(), sm->n(), (DATA_TYPE)0);

  // Copy the tiles 1-to-1
  for (int64_t j = 0; j < sm->nt(); ++j) {
    for (int64_t i = 0; i < sm->mt(); ++i) {
      if (sm->tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = sm->at(i, j, sm->tileDevice(i, j));
        C->getarr().resize(tile.size());

        DATA_TYPE *src = tile.data();
        DATA_TYPE *dst = C->getarr().data();
        cudaMemcpy(dst, src, sizeof(DATA_TYPE) * tile.size(),
                   cudaMemcpyDeviceToHost);
      }
    }
  }

  // Free resources for SLATE matrix
  sm.reset();
  sm = nullptr;

  // Save pointer to CombBLAS matrix
  cm = std::move(C);
}

void DenseMat::apply(Kernel &k) {
  assert(sm && "Can only apply kernels on SLATE matrices!");

  for (int64_t j = 0; j < sm->nt(); ++j) {   // i loops over block columns
    for (int64_t i = 0; i < sm->mt(); ++i) { // j loops over block rows
      if (sm->tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = sm->at(i, j, sm->tileDevice(i, j));
        DATA_TYPE *tile_buf = tile.data();
        k.f(tile_buf);
      }
    }
  }
}

void DenseMat::print(std::string prefix, std::ostream &out) {
  assert((sm || cm) && "Must have a SLATE or CombBLAS matrix to print!");

  // Try printing SLATE matrix
  if (sm != nullptr) {
    slate::print(prefix.c_str(), *sm);
    return;
  }

  // Try printing CombBLAS matrix
  if (cm != nullptr) {
    cm->PrintToFile("out/" + prefix);
    return;
  }
}

slate::Matrix<DATA_TYPE> *slate_gemm_(slate::Matrix<DATA_TYPE> *L,
                                      slate::Matrix<DATA_TYPE> *R, int64_t mb,
                                      int64_t nb, int64_t p, MPI_Comm comm) {
  auto B = new slate::Matrix<DATA_TYPE>(L->m(), R->n(), mb, nb, p, p, comm);
  B->insertLocalTiles(slate::Target::Devices);
  slate::gemm<DATA_TYPE>(1.0f, *L, *R, 0.0f, *B,
                         {{slate::Option::Target, slate::Target::Devices}});

  return B;
}

DenseMat DenseMat::gemm(DenseMat &R) {
  assert(sm && R.sm && "Can only do gemm on SLATE matrices!");

  std::unique_ptr<slate::Matrix<DATA_TYPE>> M =
      std::make_unique<slate::Matrix<DATA_TYPE>>(rows, R.cols, rows_per_block,
                                                 R.cols_per_block, grid_dim,
                                                 grid_dim, comm);

  M->insertLocalTiles(slate::Target::Devices);
  slate::gemm<DATA_TYPE>(1.0f, *sm, *R.sm, 0.0f, *M,
                         {{slate::Option::Target, slate::Target::Devices}});

  return DenseMat(std::move(M));
}

} // namespace popcorn
