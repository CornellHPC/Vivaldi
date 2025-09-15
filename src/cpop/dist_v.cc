#include "dist_v.hh"
#include <cuda_runtime.h>
#include <cusparse.h>

namespace cpop
{


DistV2D::DistV2D(std::vector<int>& row_ids, std::vector<int>& col_ids,
                     std::vector<float>& vals, int64_t rows, int64_t cols,
                     MPI_Comm comm, Handle& handle)
{
  // Initialize device pointers to nullptr for safe cleanup
  this->d_values = nullptr;
  this->d_colinds = nullptr;
  this->d_rowptrs = nullptr;
  this->csr_mat = nullptr;

  this->world = comm;
  MPI_Comm_size(this->world, &(this->world_size));
  MPI_Comm_rank(this->world, &(this->world_rank));

  this->grid_dim = square_grid_dim(comm);
  this->rows = rows;
  this->cols = cols;

  int col_color = this->world_rank / this->grid_dim;
  this->col_rank = this->world_rank % this->grid_dim;
  MPI_Comm_split(comm, col_color, this->col_rank, &(this->col_world));
  MPI_Comm_split(comm, this->col_rank, col_color, &(this->row_world));

  this->row_rank = col_color;

  assert( this->local_cols % cols == 0 && "Edge case not implemented yet");
  
  this->local_rows = this->rows / this->grid_dim;
  this->local_cols = this->cols / this->grid_dim;

  if (this->col_rank == this->grid_dim-1)
  {
    int64_t remainder = this->rows - ( (this->rows / this->grid_dim) * this->local_rows);
    this->local_rows += remainder;
  }

  // Distribute entries

  int * sendcounts = new int[this->world_size];
  memset(sendcounts, 0, sizeof(int) * this->world_size);
  int * senddispls = new int[this->world_size];
  memset(senddispls, 0, sizeof(int) * this->world_size);

  int64_t send_size;
  int64_t n = vals.size();
  for (int64_t i=0; i<n; i++)
  {
      float rid = row_ids[i];
      float cid = col_ids[i];

      int owner = this->map2d(rid, cid);
      sendcounts[owner]++;
      send_size++;
  }

  int * recvcounts = new int[this->world_size];
  int * recvdispls = new int[this->world_size];
  memset(recvdispls, 0, sizeof(int) * this->world_size);

  MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, this->world);

  int64_t recv_size;
  for (int i=1; i<this->world_size; i++)
  {
      senddispls[i] = sendcounts[i-1] + senddispls[i-1];
      recvdispls[i] = recvcounts[i-1] + recvdispls[i-1];
      recv_size++;
  }

  std::vector<float> values(recv_size);
  std::vector<int> colinds(recv_size);
  std::vector<int> rowinds(recv_size);

  MPI_Alltoallv(vals.data(), sendcounts, senddispls, MPI_FLOAT,
                values.data(), recvcounts, recvdispls, MPI_FLOAT,
                this->world);
  MPI_Alltoallv(col_ids.data(), sendcounts, senddispls, MPI_INT,
                colinds.data(), recvcounts, recvdispls, MPI_INT,
                this->world);
  MPI_Alltoallv(row_ids.data(), sendcounts, senddispls, MPI_INT,
                rowinds.data(), recvcounts, recvdispls, MPI_INT,
                this->world);

  // Convert COO to CSR and store on device
  convertCOOtoCSR(rowinds, colinds, values, recv_size, handle);

}

void DistV2D::convertCOOtoCSR(const std::vector<int>& coo_row,
                                const std::vector<int>& coo_col,
                                const std::vector<float>& coo_val,
                                int64_t nnz, Handle& handle)
{

  // Allocate device memory for COO format
  int *d_coo_row, *d_coo_col;
  float *d_coo_val;

  cudaMalloc(&d_coo_row, nnz * sizeof(int));
  cudaMalloc(&d_coo_col, nnz * sizeof(int));
  cudaMalloc(&d_coo_val, nnz * sizeof(float));

  // Copy COO data to device
  cudaMemcpy(d_coo_row, coo_row.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_coo_col, coo_col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_coo_val, coo_val.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);

  // Allocate device memory for CSR format (already declared as class members)
  cudaMalloc(&this->d_values, nnz * sizeof(float));
  cudaMalloc(&this->d_colinds, nnz * sizeof(int));
  cudaMalloc(&this->d_rowptrs, (this->local_rows + 1) * sizeof(int));

  // Convert COO to CSR using cuSPARSE
  cusparseXcoo2csr(handle.sh(), d_coo_row, nnz, this->local_rows, this->d_rowptrs, CUSPARSE_INDEX_BASE_ZERO);

  // Copy column indices and values (they remain the same in CSR)
  cudaMemcpy(this->d_colinds, d_coo_col, nnz * sizeof(int), cudaMemcpyDeviceToDevice);
  cudaMemcpy(this->d_values, d_coo_val, nnz * sizeof(float), cudaMemcpyDeviceToDevice);

  // Store local nnz
  this->local_nnz = nnz;

  // Create cusparseSpMat CSR matrix descriptor
  cusparseCreateCsr(&this->csr_mat,
                    this->local_rows,     // number of rows
                    this->local_cols,     // number of columns
                    nnz,                  // number of non-zeros
                    this->d_rowptrs,      // CSR row pointers
                    this->d_colinds,      // CSR column indices
                    this->d_values,       // CSR values
                    CUSPARSE_INDEX_32I,   // row pointer type
                    CUSPARSE_INDEX_32I,   // column index type
                    CUSPARSE_INDEX_BASE_ZERO, // index base
                    CUDA_R_32F);          // data type for values

  // Clean up temporary device memory
  cudaFree(d_coo_row);
  cudaFree(d_coo_col);
  cudaFree(d_coo_val);
}

DistV2D::~DistV2D()
{
  // Destroy cuSPARSE matrix descriptor
  if (this->csr_mat != nullptr) {
    cusparseDestroySpMat(this->csr_mat);
  }

  // Free device memory
  if (this->d_values != nullptr) {
    cudaFree(this->d_values);
  }
  if (this->d_colinds != nullptr) {
    cudaFree(this->d_colinds);
  }
  if (this->d_rowptrs != nullptr) {
    cudaFree(this->d_rowptrs);
  }
}

int DistV2D::map2d(float i, float j)
{

    // Assumes column major grid ordering, which is the default in SLATE
    int col_contrib = (int) (j / this->local_cols) * this->grid_dim;
    int row_contrib = (int) (i / this->local_rows);
    return col_contrib + row_contrib;
    

}


DistV2D DistV2D::initialize_v(Handle& handle, int64_t points, int64_t k, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  int grid_dim = square_grid_dim(comm);
  int P = grid_dim * grid_dim;

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
  std::vector<int> lrow_ids, lcol_ids;
  std::vector<float> lvals;
  for (int row = row_start, col = row_start % k; row < row_end;
       ++row, col = (col + 1) % k) {
    lrow_ids.push_back(row);
    lcol_ids.push_back(col);
    lvals.push_back(1.0f / points_per_cluster[col]);
  }

  // Clean up
  free(points_per_cluster);
  free(points_per_process);

  DistV2D out = DistV2D(lrow_ids, lcol_ids, lvals, k, points, comm, handle);
  return out;
}


void DistV2D::save_assignments(const char* filename) {
  // Compute global offsets
}


}
