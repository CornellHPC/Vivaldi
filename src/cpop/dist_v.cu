#include "dist_v.hh"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cassert>
#include <cstring>
#include <fstream>

namespace cpop
{

ProcessGrid::ProcessGrid(int p, int q, bool colmaj)
{
    world_size = p*q;
    row_size = p;
    col_size = q;
    world_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(world_comm, &world_rank);

    if (colmaj)
    {
        row_rank = world_rank / q;
        col_rank = world_rank % q;
    }
    else
    {
        row_rank = world_rank % p;
        col_rank = world_rank / p;
    }
    MPI_Comm_split(world_comm, row_rank, col_rank, &col_comm);
    MPI_Comm_split(world_comm, col_rank, row_rank, &row_comm);
}

DistV2D::DistV2D(int64_t m, int64_t k, std::shared_ptr<ProcessGrid> grid)
{
    this->grid = grid;
    global_cols = m;
    global_rows = k;
    global_nnz = m;

    tile_rows = compute_tile_sizes(k, grid->col_size);
    tile_cols = compute_tile_sizes(m, grid->row_size);
    tile_nnz = new int64_t[grid->world_size];
    memset(tile_nnz, 0, sizeof(int64_t) * grid->world_size);

    int64_t * row_sizes = new int64_t[k];
    memset(row_sizes, 0, sizeof(int64_t) * k);

    // Round robin assignment
    for (int64_t i=0; i<m; i++)
    {
        int64_t j = i % k;
        int owner = map2d(j, i);
        tile_nnz[owner]++;
        row_sizes[j]++;
    }
    nnz = tile_nnz[grid->world_rank];
    rows = tile_rows[grid->col_rank];
    cols = tile_cols[grid->row_rank];


    float * h_values = new float[nnz];
    int * h_rowinds = new int[nnz];
    int * h_colptrs = new int[cols+1];
    memset(h_colptrs, 0, sizeof(int) * (cols+1));

    int64_t offset = 0;
    for (int64_t i=0; i<m; i++)
    {
        int64_t j = i % k;
        int owner = map2d(j, i);
        if (owner == grid->world_rank)
        {
            h_rowinds[offset] = j % tile_rows[0];
            h_colptrs[i%cols + 1] = 1;
            h_values[offset++] = 1.0/(float)row_sizes[j];
        }
    }

    assert(offset==nnz);

    CHECK_CUDA(cudaMalloc(&d_cluster_sizes, sizeof(int) * k));
    CHECK_CUDA(cudaMemcpy(d_cluster_sizes, row_sizes, sizeof(int) * k,
                            cudaMemcpyHostToDevice));

    for (int i=1; i<cols+1; i++)
    {
        h_colptrs[i] = h_colptrs[i-1] + h_colptrs[i];
    }

    assert(h_colptrs[cols]==nnz);

    //std::ofstream ofs;
    //ofs.open(std::string("log_rank_")+std::to_string(grid->world_rank)+".out");
    //for (int i=0; i<cols+1; i++)
    //{
    //    ofs<<h_colptrs[i]<<std::endl;
    //}
    //ofs.close();

    // CSC init
    int64_t max_nnz = cols; // At most 1 nnz per column
    CHECK_CUDA(cudaMalloc(&d_colptrs, sizeof(int) * (cols+1)));
    CHECK_CUDA(cudaMalloc(&d_rowinds, sizeof(int) * max_nnz)); // avoid reallocating later
    CHECK_CUDA(cudaMalloc(&d_vals, sizeof(float) * max_nnz));

    CHECK_CUDA(cudaMemcpy(d_colptrs, h_colptrs, sizeof(int) * (cols+1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowinds, h_rowinds, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, h_values, sizeof(float) * nnz, cudaMemcpyHostToDevice));

    this->init_cusparse_csc();

    CHECK_CUDA(cudaMalloc(&d_minpairs, sizeof(FloatI32) * cols));
    CHECK_CUDA(cudaMalloc(&d_mininds, sizeof(int) * cols));

    delete[] h_colptrs;
    delete[] h_rowinds;
    delete[] h_values;
    delete[] row_sizes;
}

DistV2D::DistV2D(int64_t m, int64_t k, int loc_k, int64_t nnz, std::shared_ptr<ProcessGrid> grid):
    global_cols(m), global_rows(k), grid(grid), nnz(nnz), rows(loc_k)
{
    MPI_Allreduce(&nnz, &global_nnz, 1, MPI_INT64_T, MPI_SUM, grid->world_comm);

    tile_rows = new int[grid->col_size];
    tile_rows[grid->col_rank] = loc_k;
    MPI_Allgather(&loc_k, 1, MPI_INT, tile_rows, 1, MPI_INT, grid->col_comm);
    tile_cols = compute_tile_sizes(m, grid->row_size);


    tile_nnz = new int64_t[grid->world_size];
    memset(tile_nnz, 0, sizeof(int64_t) * grid->world_size);
    tile_nnz[grid->world_rank] = nnz;

    MPI_Allreduce(MPI_IN_PLACE, tile_nnz, grid->world_size, MPI_INT64_T, MPI_SUM, grid->world_comm);

    //rows = tile_rows[grid->col_rank];
    cols = tile_cols[grid->row_rank];

    CHECK_CUDA(cudaMalloc(&d_vals, sizeof(float) * nnz));
    CHECK_CUDA(cudaMalloc(&d_rowinds, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc(&d_colptrs, sizeof(int) * (cols+1)));
    CHECK_CUDA(cudaMemset(d_colptrs, 0, sizeof(int)))

    CHECK_CUDA(cudaMalloc(&d_remote_vals, sizeof(float) * cols));
    CHECK_CUDA(cudaMalloc(&d_remote_rowinds, sizeof(int) * cols));
    CHECK_CUDA(cudaMalloc(&d_remote_colptrs, sizeof(int) * (cols+1)));
    CHECK_CUDA(cudaMemset(d_remote_colptrs, 0, sizeof(int)))

    CHECK_CUDA(cudaMalloc(&d_cluster_sizes, sizeof(int) * k));
    CHECK_CUDA(cudaMalloc(&d_minpairs, sizeof(FloatI32) * cols));
    CHECK_CUDA(cudaMalloc(&d_mininds, sizeof(int) * cols));

    CHECK_CUDA(cudaMalloc(&d_v_dense, sizeof(float) * loc_k * cols));

    this->init_cusparse_csc();

}

void DistV2D::init_cusparse_csc()
{
    CHECK_CUSPARSE(cusparseCreateCsc(
                      &(this->csc_mat),
                      this->rows,     // number of rows
                      this->cols,     // number of columns
                      this->nnz,                  // number of non-zeros
                      this->d_colptrs,      // CSC column pointers
                      this->d_rowinds,      // CSC row indices
                      this->d_vals,         // CSC values
                      CUSPARSE_INDEX_32I,   // column pointer type
                      CUSPARSE_INDEX_32I,   // row index type
                      CUSPARSE_INDEX_BASE_ZERO, // index base
                      CUDA_R_32F));          // data type for values
}

void DistV2D::update_nnz(int64_t nnz)
{
    this->nnz = nnz;
    memset(this->tile_nnz, 0, sizeof(int64_t) * grid->world_size);
    this->tile_nnz[grid->world_rank] = nnz;
    MPI_Allreduce(MPI_IN_PLACE, this->tile_nnz, this->grid->world_size, MPI_INT64_T, MPI_SUM, this->grid->world_comm);
}

DistV2D::~DistV2D()
{
    // Destroy cuSPARSE matrix descriptor
    if (this->csc_mat != nullptr) {
        cusparseDestroySpMat(this->csc_mat);
    }

    // Free device memory
    if (this->d_vals != nullptr) {
        cudaFree(this->d_vals);
    }
    if (this->d_rowinds != nullptr) {
        cudaFree(this->d_rowinds);
    }
    if (this->d_colptrs != nullptr) {
        cudaFree(this->d_colptrs);
    }
    if (this->d_cluster_sizes!= nullptr) {
        cudaFree(this->d_cluster_sizes);
    }
    if (this->d_minpairs!= nullptr) {
        cudaFree(this->d_minpairs);
    }
    if (this->d_mininds!= nullptr) {
        cudaFree(this->d_mininds);
    }
    if (this->d_remote_vals != nullptr) {
        cudaFree(this->d_remote_vals);
    }
    if (this->d_remote_rowinds != nullptr) {
        cudaFree(this->d_remote_rowinds);
    }
    if (this->d_remote_colptrs != nullptr) {
        cudaFree(this->d_remote_colptrs);
    }
    if (this->d_v_dense != nullptr) {
        cudaFree(this->d_v_dense);
    }

    delete[] tile_rows;
    delete[] tile_cols;
    delete[] tile_nnz;
}

int DistV2D::map2d(int64_t rid, int64_t cid)
{
    int row_contrib = (rid / tile_rows[0] );
    int col_contrib = (cid / tile_cols[0] ) * grid->col_size;
    return col_contrib + row_contrib;
}

}
