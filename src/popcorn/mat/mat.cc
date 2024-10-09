#include "../utils/utils.hh"
#include "dense_mat.hh"
#include "sparse_mat.hh"

namespace popcorn {

void fill_slate_mat_with_buffer(slate::Matrix<DATA_TYPE>* M, DATA_TYPE* buf) {
  int64_t m = M->m(), n = M->n(), mb = M->tileMb(0), nb = M->tileNb(0);
  for (int64_t j = 0; j < M->nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < M->mt(); ++i) {  // j loops over block rows
      if (M->tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = M->at(i, j, M->tileDevice(i, j));
        int64_t lda = tile.stride();
        DATA_TYPE* A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {  // jj loops over columns
          int64_t global_column = j * nb + jj;
          int64_t global_row_start = i * mb;
          cudaMemcpy(A + jj * lda, buf + global_column * m + global_row_start,
                     sizeof(DATA_TYPE) * tile.mb(), cudaMemcpyHostToDevice);
        }
      }
    }
  }
}

DenseMat::~DenseMat() {
  free(sm);
  free(cm);
}

DenseMat DenseMat::load_from_file(const char* filename, int64_t rows,
                                  int64_t cols, int64_t rows_per_block,
                                  int64_t cols_per_block, int64_t grid_dim,
                                  MPI_Comm comm) {
  // Open file
  MPI_File fh;
  MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Read data
  size_t count = rows * cols * sizeof(DATA_TYPE);
  DATA_TYPE* data = (DATA_TYPE*) malloc(count);
  MPI_File_read_all(fh, data, rows * cols, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Create empty SLATE matrix object of size equal to data
  auto M = new slate::Matrix<DATA_TYPE>(
      rows, cols, rows_per_block, cols_per_block, grid_dim, grid_dim, comm);
  M->insertLocalTiles(slate::Target::Devices);

  // Initialize data
  fill_slate_mat_with_buffer(M, data);

  // Clean up
  free(data);
  MPI_File_close(&fh);

  return DenseMat::from_slate(M, rows_per_block, cols_per_block, grid_dim);
}

DenseMat DenseMat::from_slate(slate::Matrix<DATA_TYPE>* sm,
                              int64_t rows_per_block, int64_t cols_per_block,
                              int64_t grid_dim) {
  return DenseMat(sm->m(), sm->n(), sm->mt(), sm->nt(), rows_per_block,
                  cols_per_block, grid_dim, sm->mpiComm(), sm, NULL);
}

void DenseMat::apply(Kernel& k) {
  assert(sm && "Can only apply kernels on SLATE matrices!");

  for (int64_t j = 0; j < sm->nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < sm->mt(); ++i) {  // j loops over block rows
      if (sm->tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = sm->at(i, j, sm->tileDevice(i, j));
        DATA_TYPE* tile_buf = tile.data();
        k.f(tile_buf);
      }
    }
  }
}

void DenseMat::print(std::ostream& out, std::string prefix) {
  assert(sm && "Can only print SLATE metrics!");
  slate::print(prefix.c_str(), *sm);
}

slate::Matrix<DATA_TYPE>* slate_gemm_(slate::Matrix<DATA_TYPE>* L,
                                      slate::Matrix<DATA_TYPE>* R, int64_t mb,
                                      int64_t nb, int64_t p, MPI_Comm comm) {
  auto B = new slate::Matrix<DATA_TYPE>(L->m(), R->n(), mb, nb, p, p, comm);
  B->insertLocalTiles(slate::Target::Devices);
  slate::gemm<DATA_TYPE>(1.0f, *L, *R, 0.0f, *B,
                         {{slate::Option::Target, slate::Target::Devices}});
  return B;
}

DenseMat DenseMat::symmetric_product() {
  assert(sm && "Can only do gemm on SLATE matrices!");
  auto transposed = slate::transpose(*sm);
  auto B = slate_gemm_(sm, &transposed, rows_per_block, rows_per_block,
                       grid_dim, comm);
  return DenseMat::from_slate(B, rows_per_block, rows_per_block, grid_dim);
}

DenseMat gemm(DenseMat& L, DenseMat& R) {
  assert(L.sm && R.sm && "Can only do gemm on SLATE matrices!");
  auto B = slate_gemm_(L.sm, R.sm, L.rows_per_block, R.cols_per_block,
                       L.grid_dim, L.comm);
  return DenseMat::from_slate(B, L.rows_per_block, R.cols_per_block,
                              L.grid_dim);
}

SparseMat::SparseMat(std::vector<float>& row_ids,
                     std::vector<float>& col_ids,
                     std::vector<DATA_TYPE>& vals, int64_t rows, int64_t cols,
                     int64_t grid_dim, MPI_Comm comm) {
  this->grid_dim = grid_dim;
  this->comm = comm;

  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(comm, grid_dim, grid_dim);

  combblas::FullyDistVec<int64_t, DATA_TYPE> drows(row_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dcols(col_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dvals(vals, grid);

  combblas::SpParMat<int64_t, DATA_TYPE, UDER> V{rows,  cols,  drows,
                                                 dcols, dvals, false};
  this->cm = V;
}

}  // namespace popcorn
