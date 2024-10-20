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

std::vector<DATA_TYPE> DenseMat::initialize_z(ClusterAssignment &assignment,
                                              DenseMat &ET) {
  assert(ET.cm != nullptr && "ET must be a dense CombBLAS matrix!");
  // std::cout << ET.comm << std::endl;

  // Compute communicator information
  int rank, size, grid_dim, row_index, col_index;
  MPI_Comm_rank(ET.comm, &rank);
  MPI_Comm_size(ET.comm, &size);
  grid_dim = square_grid_dim(ET.comm);
  row_index = size / grid_dim;
  col_index = size % grid_dim;

  // Prepare for MPI
  long mb = ET.cm->getnrow();
  long nb = ET.cm->getncol();
  long *rows = new long[size];
  long *cols = new long[size];

  // Patch weird local size reporting
  if (mb == 0 or nb == 0) {
    mb = 0;
    nb = 0;
  }

  // Run communication
  MPI_Allgather(&mb, 1, MPI_LONG, rows, 1, MPI_LONG, ET.comm);
  MPI_Allgather(&nb, 1, MPI_LONG, cols, 1, MPI_LONG, ET.comm);

  // Compute row range
  int row_start = 0;
  for (int i = rank % grid_dim; i < rank; i += grid_dim) {
    row_start += rows[i];
  }
  int row_end = row_start + rows[rank];

  // Compute col range
  int col_start = 0;
  for (int i = rank - (rank % grid_dim); i < rank; ++i) {
    col_start += cols[i];
  }
  int col_end = col_start + cols[rank];

  // Clean up matrix data
  delete[] rows;
  delete[] cols;

  std::vector<float> points = assignment.get_points();
  std::vector<float> clusters = assignment.get_clusters();
  std::vector<DATA_TYPE> lvals;
  std::vector<float> lcols;
  std::vector<DATA_TYPE> arr = ET.cm->getarr();

  // Find local assignments
  for (int i = 0; i < points.size(); ++i) {
    float point = points.at(i);
    float cluster = clusters.at(i);

    if (row_start <= cluster && cluster < row_end && col_start <= point &&
        point < col_end) {
      int row = cluster - row_start;
      int col = point - col_start;

      lvals.push_back(arr.at(row * nb + col));
      lcols.push_back(point);
    }
  }

  // MPI initialization
  int *recvcounts = new int[size];
  int *displs = new int[size];
  DATA_TYPE *zvals = new DATA_TYPE[assignment.get_total_points()];
  float *zcols = new float[assignment.get_total_points()];
  int elems = lvals.size();

  // Determine dynamic message sizes
  MPI_Allgather(&elems, 1, MPI_INT, recvcounts, 1, MPI_INT, ET.comm);
  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }

  // MPI communication (TODO: dynamically determine MPI type)
  MPI_Allgatherv(lvals.data(), lvals.size(), MPI_FLOAT, zvals, recvcounts,
                 displs, MPI_FLOAT, ET.comm);
  MPI_Allgatherv(lcols.data(), lcols.size(), MPI_FLOAT, zcols, recvcounts,
                 displs, MPI_FLOAT, ET.comm);

  // Points should be approximately sorted on column, so
  // insertion sort should be approximately linear time
  int length = assignment.get_total_points();
  for (int i = 1; i < length; ++i) {
    for (int j = i - 1; j >= 0; --j) {
      if (zcols[j] > zcols[j + 1]) {
        float tcol = zcols[j];
        zcols[j] = zcols[j + 1];
        zcols[j + 1] = tcol;

        DATA_TYPE tval = zvals[j];
        zvals[j] = zvals[j + 1];
        zvals[j + 1] = tval;
      } else {
        break;
      }
    }
  }

  // Copy heap data to stack (pass via RVO)
  std::vector<DATA_TYPE> out(zvals, zvals + length);

  // Clean up
  delete[] recvcounts;
  delete[] displs;
  delete[] zvals;
  delete[] zcols;

  return out;
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

DenseMat::DenseMat(std::unique_ptr<combblas::DnParMat<int64_t, DATA_TYPE>> M,
                   MPI_Comm comm) {
  // Save pointer to matrix
  cm = std::move(M);

  // Populate private fields
  this->rows = cm->getgnrow();
  this->cols = cm->getgncol();
  this->comm = comm;
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
        k.f(tile.mb(), tile.nb(), tile_buf);
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
