#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>

#include "cuda_runtime.h"
#include "mpi.h"
#include "mpio.h"

#include "cluster.hh"
#include "utils.hh"
#include "gpu_kernels.cuh"
#include <cub/cub.cuh>


namespace cpop {

V_t::V_t(int64_t m, int64_t k, bool sparse, MPI_Comm comm) {
  // Struct member initialization
  m_ = m;
  k_ = k;
  this->sparse = sparse;
  this->comm = comm;

  // MPI initializations
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_procs);
  t_sizes = compute_tile_sizes(m, n_procs);
  t = t_sizes[rank];

  displs = (int*)calloc(n_procs, sizeof(int));
  for (int i = 1; i < n_procs; ++i)
    displs[i] = displs[i - 1] + t_sizes[i - 1];  // MPI displacements

  // Struct data initialization
  CHECK_CUDA(cudaMalloc(&global_assignments, m * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&global_cluster_sizes, k * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&local_assignments, t * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&local_cluster_sizes, k * sizeof(int)));

  // round robin initialization (todo: GPU)
  int* init_global_cluster_sizes = (int*)calloc(k, sizeof(int));
  for (int i = 0; i < k; ++i)
    init_global_cluster_sizes[i] = (m / k) + ((i < m % k) ? 1 : 0);
  CHECK_CUDA(cudaMemcpy(global_cluster_sizes, init_global_cluster_sizes,
                        k * sizeof(int), cudaMemcpyHostToDevice));

  // round robin initialization (todo: GPU)
  int* init_assignments = (int*)calloc(m, sizeof(int));
  for (int i = 0; i < m; ++i)
    init_assignments[i] = i % k;
  CHECK_CUDA(cudaMemcpy(global_assignments, init_assignments, m * sizeof(int),
                        cudaMemcpyHostToDevice));

  // set assignment pointers
  local_ptr_to_assignments = global_assignments + displs[rank];
  previous_global_assignments = nullptr;
  CHECK_CUDA(cudaMalloc(&converged, sizeof(bool)));
  CHECK_CUDA(cudaMalloc(&local_k_means_objective_score, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&local_k_means_objective_delta, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&prev_point_to_cluster_distances,
                        t * sizeof(float)));  // for convergence checking
  CHECK_CUDA(cudaMemset(prev_point_to_cluster_distances, 0,
                        t * sizeof(float)));  // initialize to zero
  previous_global_k_means_objective_score =
      1e-6f;  // a very small number to start
  previous_local_k_means_objective_score =
      1e-6f;  // a very small number to start
              
  CHECK_CUDA(cudaMalloc(&d_csr_colinds, sizeof(int) * (m)));
  CHECK_CUDA(cudaMalloc(&d_csr_rowptrs, sizeof(int) * (k + 1)));
  CHECK_CUDA(cudaMalloc(&d_csr_val, sizeof(float) * (m)));

  // Implementation-specific initialization
  if (sparse) {
    CHECK_CUDA(cudaMalloc(&values, m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&global_csc_col_offsets, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&local_csc_col_offsets, (t + 1) * sizeof(int)));


    // basic CSC initializations (todo: GPU)
    int* global_csc_col_offsets_ = (int*)calloc(m + 1, sizeof(int));
    for (int i = 0; i < m; ++i)
      global_csc_col_offsets_[i + 1] = i + 1;
    CHECK_CUDA(cudaMemcpy(global_csc_col_offsets, global_csc_col_offsets_,
                          (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(local_csc_col_offsets, global_csc_col_offsets_,
                          (t + 1) * sizeof(int), cudaMemcpyHostToDevice));
    free(global_csc_col_offsets_);

    // round robin initialization (todo: GPU)
    float* init_values = (float*)calloc(m, sizeof(float));
    for (int i = 0; i < m; ++i)
      init_values[i] = 1.0f / init_global_cluster_sizes[i % k];
    CHECK_CUDA(cudaMemcpy(values, init_values, m * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(init_values);

    // cusparse initializations
    CHECK_CUSPARSE(cusparseCreateCsc(&gV, k, m, m, global_csc_col_offsets,
                                     global_assignments, values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // the local partition of V in CSC is found by slicing the global partition at [displs[rank]:displs[rank] + t]
    // (since displs[rank] is the displacement of this rank's first point)
    // this is done with simple pointer arithmetic
    local_ptr_to_values = values + displs[rank];
    CHECK_CUSPARSE(cusparseCreateCsc(
        &lV, k, t, t, local_csc_col_offsets, local_ptr_to_assignments,
        local_ptr_to_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  } else {
    // round robin initialization (TODO: gpu)
    float* values = (float*)calloc(k * m, sizeof(float));
    for (int64_t cluster = 0; cluster < k; ++cluster) {
      for (int64_t point = cluster; point < m; point += k) {
        int64_t total = m / k + ((cluster < m % k) ? 1 : 0);
        values[cluster * m + point] = 1.0f / total;
      }
    }
    CHECK_CUDA(cudaMalloc(&this->values, k * m * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(this->values, values, k * m * sizeof(float),
                          cudaMemcpyHostToDevice));
    local_ptr_to_values = this->values + displs[rank];
    free(values);
  }

  //CHECK_CUDA(cudaMalloc(&d_v_dense, sizeof(float) * m * k * (int)std::floor(std::sqrt(n_procs))));

  // Clean up
  free(init_global_cluster_sizes);
  free(init_assignments);
}


int V_t::save(const char* path) {
  MPI_File fh;
  MPI_File_open(comm, path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                &fh);

  // Compute offset for rank
  int offset = displs[rank];

  int* assignments = (int*)malloc(t * sizeof(int));
  CHECK_CUDA(cudaMemcpy(assignments, local_ptr_to_assignments, t * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // Write the data to disk
  MPI_File_write_at(fh, offset * sizeof(int), assignments, t, MPI_INT,
                    MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
  free(assignments);
  return EXIT_SUCCESS;
}

void V_t::print() {
  std::cout << "Printing V from rank " << rank << std::endl;

  if (sparse) {
    std::cout << "Local displacement is " << displs[rank] << std::endl;
    std::cout << "Global device vectors:" << std::endl;
    print_device_buffer(global_assignments, m_);
    print_device_buffer(global_csc_col_offsets, m_ + 1);
    print_device_buffer(values, m_);
    std::cout << "Local device vectors:" << std::endl;
    print_device_buffer(local_ptr_to_assignments, t);
    print_device_buffer(local_csc_col_offsets, t + 1);
    print_device_buffer(local_ptr_to_values, t);
    std::cout << "Working vectors:" << std::endl;
    print_device_buffer(global_cluster_sizes, k_);
    print_device_buffer(local_assignments, t);
    print_device_buffer(local_cluster_sizes, k_);
  } else {
    print_device_matrix(values, k_, m_);
  }

  std::cout << "-------------------" << std::endl;
}

bool V_t::test_convergence() {
  if (previous_global_assignments &&
      test_convergence_equality(global_assignments, previous_global_assignments,
                                m_)) {
    // Free everything and return true because we have converged
    CHECK_CUDA(cudaFree(previous_global_assignments));
    previous_global_assignments = nullptr;
    return true;  // Converged
  } else if (!previous_global_assignments) {
    // First iteration, so we need to allocate previous_global_assignments
    CHECK_CUDA(cudaMalloc(&previous_global_assignments, m_ * sizeof(int)));
  }
  // Set previous_global_assignments to current global_assignments
  cudaMemcpy(previous_global_assignments, global_assignments, m_ * sizeof(int),
             cudaMemcpyDeviceToDevice);
  return false;  // Not converged
}

V_t::~V_t() {
  CHECK_CUDA(cudaFree(converged));
  CHECK_CUDA(cudaFree(local_k_means_objective_score));
  CHECK_CUDA(cudaFree(local_k_means_objective_delta));
  CHECK_CUDA(cudaFree(prev_point_to_cluster_distances));
  CHECK_CUDA(cudaFree(d_csr_colinds));
  CHECK_CUDA(cudaFree(d_csr_rowptrs));
  CHECK_CUDA(cudaFree(d_csr_val));
  free(t_sizes);
  free(displs);
  if (sparse) {
    CHECK_CUDA(cudaFree(global_assignments));
    CHECK_CUDA(cudaFree(global_cluster_sizes));
    CHECK_CUDA(cudaFree(local_assignments));
    CHECK_CUDA(cudaFree(local_cluster_sizes));
    CHECK_CUDA(cudaFree(values));
    CHECK_CUDA(cudaFree(global_csc_col_offsets));
    CHECK_CUDA(cudaFree(local_csc_col_offsets));
    CHECK_CUSPARSE(cusparseDestroySpMat(gV));
    CHECK_CUSPARSE(cusparseDestroySpMat(lV));
    if (previous_global_assignments)
      CHECK_CUDA(cudaFree(previous_global_assignments));
  } else {
    CHECK_CUDA(cudaFree(values));
  }
}


DistV1D::DistV1D(int64_t m, int64_t k, bool sparse, std::shared_ptr<ProcessGrid> grid)
{
    this->local_v = new V_t(m, k, true, grid->world_comm);
    this->grid = grid;

    int sqrtp = grid->row_size;

    CHECK_CUDA(cudaMalloc(&this->d_remote_vals, sizeof(float) * sqrtp * local_v->t));
    CHECK_CUDA(cudaMalloc(&this->d_remote_rowinds, sizeof(int) * sqrtp * local_v->t));
    CHECK_CUDA(cudaMalloc(&this->d_remote_colptrs, sizeof(int) * (sqrtp * local_v->t + 1)));
    CHECK_CUDA(cudaMemset(this->d_remote_colptrs,0,sizeof(int)*(sqrtp*local_v->t + 1)));

    CHECK_CUSPARSE(cusparseCreateCsc(&v_cusparse,
                                     local_v->k_,
                                     sqrtp*local_v->t,
                                     sqrtp*local_v->t,
                                     d_remote_colptrs,
                                     d_remote_rowinds,
                                     d_remote_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    if (!sparse)
    {
        CHECK_CUDA(cudaMalloc(&d_v_dense, sizeof(float) * local_v->t * k * sqrtp));
    }
    else
    {
        d_v_dense = nullptr;
    }

}

DistV1D::~DistV1D()
{
    if (this->d_remote_vals != nullptr)
    {
        CHECK_CUDA(cudaFree(this->d_remote_vals));
    }
    if (this->d_remote_rowinds != nullptr)
    {
        CHECK_CUDA(cudaFree(this->d_remote_rowinds));
    }
    if (this->d_remote_colptrs != nullptr)
    {
        CHECK_CUDA(cudaFree(this->d_remote_colptrs));
    }
    if (this->d_v_dense != nullptr)
    {
        CHECK_CUDA(cudaFree(this->d_v_dense));
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(v_cusparse));
}


DnMat_t::DnMat_t(int64_t h, int64_t w) {
  CHECK_CUDA(cudaMalloc(&dM, h * w * sizeof(float)));
  CHECK_CUSPARSE(
      cusparseCreateDnMat(&M, h, w, w, dM, CUDA_R_32F, CUSPARSE_ORDER_ROW));

  h_ = h;
  w_ = w;
}

DnMat_t::DnMat_t(int64_t h, int64_t w, float* dM_) {
  dM = dM_;
  CHECK_CUSPARSE(
      cusparseCreateDnMat(&M, h, w, w, dM, CUDA_R_32F, CUSPARSE_ORDER_ROW));

  h_ = h;
  w_ = w;
}

int DnMat_t::print() {
  print_device_matrix(dM, h_, w_);
  return EXIT_SUCCESS;
}

DnMat_t::~DnMat_t() {
  CHECK_CUDA(cudaFree(dM));
  CHECK_CUSPARSE(cusparseDestroyDnMat(M));
}

DnVec_t::DnVec_t(int t) {
  CHECK_CUDA(cudaMalloc(&dz, t * sizeof(float)));
  CHECK_CUSPARSE(cusparseCreateDnVec(&z, t, dz, CUDA_R_32F));
  size = t;
}

int DnVec_t::print() {
  print_device_buffer(dz, size);
  return EXIT_SUCCESS;
}

DnVec_t::~DnVec_t() {
  CHECK_CUDA(cudaFree(dz));
  CHECK_CUSPARSE(cusparseDestroyDnVec(z));
}

void csc_to_csr(Handle& handle, cusparseSpMatDescr_t * spmat, 
                int64_t rows, int64_t cols, int64_t nnz,
                float * d_csc_val, int * d_csc_rowinds, int * d_csc_colptrs,
                float * d_csr_val, int * d_csr_colinds, int * d_csr_rowptrs)
{
    size_t buf_size = 0;
    void * d_buf = nullptr;
    CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(handle.sh(),
                                                 cols,
                                                 rows,
                                                 nnz,
                                                 d_csc_val,
                                                 d_csc_colptrs,
                                                 d_csc_rowinds,
                                                 d_csr_val,
                                                 d_csr_rowptrs,
                                                 d_csr_colinds,
                                                 CUDA_R_32F,
                                                 CUSPARSE_ACTION_NUMERIC,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                                 &buf_size));
    CHECK_CUDA(cudaMalloc(&d_buf, buf_size));
    CHECK_CUSPARSE(cusparseCsr2cscEx2(handle.sh(),
                                     cols,
                                     rows,
                                     nnz,
                                     d_csc_val,
                                     d_csc_colptrs,
                                     d_csc_rowinds,
                                     d_csr_val,
                                     d_csr_rowptrs,
                                     d_csr_colinds,
                                     CUDA_R_32F,
                                     CUSPARSE_ACTION_NUMERIC,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                     d_buf));

    CHECK_CUSPARSE(cusparseCreateCsr(spmat,
                                     rows,
                                     cols,
                                     nnz,
                                     d_csr_rowptrs,
                                     d_csr_colinds,
                                     d_csr_val,
                                     CUSPARSE_INDEX_32I,   
                                     CUSPARSE_INDEX_32I,   
                                     CUSPARSE_INDEX_BASE_ZERO, 
                                     CUDA_R_32F));          
    CHECK_CUDA(cudaFree(d_buf));
}

int spmm2d(Handle& handle, DistV2D& V, DistDnMat_t& K, DistDnMat_t& E)
{

    std::cerr<<"do not use spmm2d"<<std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);

    auto grid = V.grid;

    const int niters = grid->row_size;
    int64_t * recv_nnz = new int64_t[niters];
    memset(recv_nnz, 0, sizeof(int64_t) * niters);
    recv_nnz[grid->row_rank] = V.nnz;

    MPI_Allreduce(MPI_IN_PLACE, recv_nnz, niters, MPI_INT64_T, MPI_SUM, grid->row_comm);


    float * K_recv; 
    float * K_send;

    CHECK_CUDA(cudaMalloc(&K_recv, sizeof(float) * K.mat->h_ * K.mat->w_));

    float * d_vals_send;
    int * d_rowinds_send;
    int * d_colptrs_send;

    cusparseSpMatDescr_t loc_V;
    cusparseDnMatDescr_t loc_K;

    for (int i=0; i<niters; i++)
    {


        if (i == grid->col_rank)
        {
            K_send = K.mat->dM;
        }
        else
        {
            K_send = K_recv;
        }

#ifndef BASIC
        auto bcast_start = hrc::now();
#endif
        MPI_Bcast(K_send, K.mat->h_ * K.mat->w_, MPI_FLOAT, i, grid->col_comm);
#ifndef BASIC
        timer.e_mpi += get_time_elapsed(bcast_start);
#endif

        if (i == grid->row_rank)
        {
            d_vals_send = V.d_vals;
            d_rowinds_send = V.d_rowinds;
            d_colptrs_send = V.d_colptrs;
        }
        else
        {
            CHECK_CUDA(cudaMalloc(&d_vals_send, sizeof(float) * recv_nnz[i]));
            CHECK_CUDA(cudaMalloc(&d_rowinds_send, sizeof(int) * recv_nnz[i]));
            CHECK_CUDA(cudaMalloc(&d_colptrs_send, sizeof(int) * (V.tile_cols[i] + 1)));
        }
        
#ifndef BASIC
        bcast_start = hrc::now();
#endif
        MPI_Bcast(d_vals_send, recv_nnz[i], MPI_FLOAT, i, grid->row_comm);
        MPI_Bcast(d_rowinds_send, recv_nnz[i], MPI_INT, i, grid->row_comm);
        MPI_Bcast(d_colptrs_send, V.tile_cols[i] + 1, MPI_INT, i, grid->row_comm);
#ifndef BASIC
        timer.e_mpi += get_time_elapsed(bcast_start);
#endif

#ifndef BASIC
        auto comp_start = hrc::now();
#endif
        CHECK_CUSPARSE(cusparseCreateCsc(&loc_V, V.tile_rows[i], V.tile_cols[i],
                          recv_nnz[i],
                          d_colptrs_send, 
                          d_rowinds_send,
                          d_vals_send, 
                          CUSPARSE_INDEX_32I,   
                          CUSPARSE_INDEX_32I,   
                          CUSPARSE_INDEX_BASE_ZERO, 
                          CUDA_R_32F));          

        CHECK_CUSPARSE(cusparseCreateDnMat(&loc_K,
                            K.mat->h_,
                            K.mat->w_,
                            K.mat->w_,
                            K_send,
                            CUDA_R_32F,
                            CUSPARSE_ORDER_ROW));
        CHECK_CUDA(cudaDeviceSynchronize());

        float alpha = 1.0;
        float beta = (i==0) ? 0.0 : 1.0;

        // Buffer size 
        size_t buffer_size;
        void* buffer;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, loc_V, loc_K, &beta, E.mat->M,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
        CHECK_CUDA(cudaMalloc(&buffer, buffer_size));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Perform SpMM
        CHECK_CUSPARSE(cusparseSpMM(
            handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, loc_V, loc_K, &beta, E.mat->M,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Clean up
        CHECK_CUDA(cudaFree(buffer));
        CHECK_CUSPARSE(cusparseDestroySpMat(loc_V));
        CHECK_CUSPARSE(cusparseDestroyDnMat(loc_K));

#ifndef BASIC
        timer.e_spmm += get_time_elapsed(comp_start);
#endif


        if (i != grid->row_rank)
        {
            CHECK_CUDA(cudaFree(d_vals_send));
            CHECK_CUDA(cudaFree(d_rowinds_send));
            CHECK_CUDA(cudaFree(d_colptrs_send));
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);

    }

    CHECK_CUDA(cudaFree(K_recv));

    delete[] recv_nnz;
    return EXIT_SUCCESS;
}


// Do not communicate tiles of the kernel matrix 
int spmm2d_bs(Handle& handle, DistV2D& V, DistV2D& V_tr, DistDnMat_t& K, DistDnMat_t& E, float * d_T)
{

    auto grid = V.grid;
    const int niters = grid->row_size;


    float * d_vals_send;
    int * d_rowinds_send;
    int * d_colptrs_send;

    cusparseSpMatDescr_t loc_V;
    cusparseDnMatDescr_t T;

    // Transpose the sparse matrix
    int tr_rank = (grid->col_rank) * grid->col_size + grid->row_rank;
    V_tr.nnz = V.tile_nnz[tr_rank];

#ifndef BASIC
    auto preprocess = hrc::now();
#endif

    MPI_Sendrecv(V.d_vals, V.nnz, MPI_FLOAT, tr_rank, 100, 
                 V_tr.d_vals, V_tr.nnz, MPI_FLOAT, tr_rank, 100,
                 grid->world_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(V.d_rowinds, V.nnz, MPI_INT, tr_rank, 101, 
                 V_tr.d_rowinds, V_tr.nnz, MPI_INT, tr_rank, 101,
                 grid->world_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(V.d_colptrs, V.cols+1, MPI_INT, tr_rank, 102, 
                 V_tr.d_colptrs, V_tr.cols+1, MPI_INT, tr_rank, 102,
                 grid->world_comm, MPI_STATUS_IGNORE);


    int64_t * recv_nnz = new int64_t[niters];
    memset(recv_nnz, 0, sizeof(int64_t) * niters);
    recv_nnz[grid->row_rank] = V_tr.nnz;
    MPI_Allreduce(MPI_IN_PLACE, recv_nnz, niters, MPI_INT64_T, MPI_SUM, grid->row_comm);


#ifndef BASIC
    timer.e_mpi += get_time_elapsed(preprocess);
#endif


    for (int i=0; i<niters; i++)
    {
#ifdef DEBUG2D
        if (grid->world_rank==0)
        {
            std::cout<<"Iteration "<<i<<std::endl;
        }
#endif

        if (i == grid->row_rank)
        {
            d_vals_send = V_tr.d_vals;
            d_rowinds_send = V_tr.d_rowinds;
            d_colptrs_send = V_tr.d_colptrs;
        }
        else
        {
            d_vals_send = V_tr.d_remote_vals;
            d_rowinds_send = V_tr.d_remote_rowinds;
            d_colptrs_send = V_tr.d_remote_colptrs;
        }
        
#ifndef BASIC
        auto bcast_start = hrc::now();
#endif

#ifdef DEBUG2D
        if (grid->world_rank==0)
        {
            std::cout<<"bcast "<<i<<std::endl;
        }
#endif

        MPI_Bcast(d_vals_send, recv_nnz[i], MPI_FLOAT, i, grid->row_comm);
        MPI_Bcast(d_rowinds_send, recv_nnz[i], MPI_INT, i, grid->row_comm);
        MPI_Bcast(d_colptrs_send, V_tr.tile_cols[i] + 1, MPI_INT, i, grid->row_comm);

#ifndef BASIC
        CHECK_CUDA(cudaDeviceSynchronize());
        timer.e_mpi += get_time_elapsed(bcast_start);
#endif

#ifndef BASIC
        auto comp_start = hrc::now();
#endif


        CHECK_CUSPARSE(cusparseCreateCsc(&loc_V, V_tr.tile_rows[i], V_tr.tile_cols[i],
                          recv_nnz[i],
                          d_colptrs_send, 
                          d_rowinds_send,
                          d_vals_send, 
                          CUSPARSE_INDEX_32I,   
                          CUSPARSE_INDEX_32I,   
                          CUSPARSE_INDEX_BASE_ZERO, 
                          CUDA_R_32F));          
        CHECK_CUDA(cudaDeviceSynchronize());

        float alpha = 1.0f;
        float beta = 0.0f;

        bool is_sparse = (recv_nnz[i] / (V_tr.tile_rows[i] * V_tr.tile_cols[i])) <= SPARSITY_THRESH;

        if (is_sparse)
        {

            CHECK_CUSPARSE(cusparseCreateDnMat(&T, V_tr.tile_rows[i], V_tr.tile_cols[i], V_tr.tile_cols[i], d_T, CUDA_R_32F, CUSPARSE_ORDER_ROW));

            // Buffer size 
            size_t buffer_size;
            void* buffer;
            CHECK_CUSPARSE(cusparseSpMM_bufferSize(
                handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, loc_V, K.mat->M, &beta, T,
                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
            CHECK_CUDA(cudaMalloc(&buffer, buffer_size));
            CHECK_CUDA(cudaDeviceSynchronize());

            // Perform SpMM
            CHECK_CUSPARSE(cusparseSpMM(
                handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, loc_V, K.mat->M, &beta, T,
                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer));
            CHECK_CUDA(cudaDeviceSynchronize());

            // Clean up
            CHECK_CUDA(cudaFree(buffer));
            CHECK_CUSPARSE(cusparseDestroyDnMat(T));
        }
        else
        {
            assert( V_tr.global_rows % grid->col_size == 0 && "Dense 2D does not work when k % sqrt(P) != 0\n");
            cusparseDnMatDescr_t v_dense;
            size_t buf_size = 0;
            void * d_buf = nullptr;

            //CHECK_CUDA(cudaMemset(V_tr.d_v_dense, 0, sizeof(float) * V_tr.tile_rows[i] * V_tr.tile_cols[i]));
            CHECK_CUSPARSE(cusparseCreateDnMat(&v_dense, V_tr.tile_rows[i], V_tr.tile_cols[i], V_tr.tile_cols[i], V_tr.d_v_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW));
            CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(handle.sh(), loc_V, v_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &buf_size));
            CHECK_CUDA(cudaMalloc(&d_buf, buf_size));
            CHECK_CUSPARSE(cusparseSparseToDense(handle.sh(), loc_V, v_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_buf));
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaFree(d_buf));

            CHECK_CUSPARSE(cusparseDestroyDnMat(v_dense));

            CHECK_CUBLAS(cublasSgemm(handle.dh(),
                                     CUBLAS_OP_N, CUBLAS_OP_N, 
                                     V_tr.tile_cols[i],
                                     V_tr.tile_rows[i],
                                     V_tr.tile_cols[i],
                                     &alpha,
                                     K.mat->dM, V_tr.tile_cols[i],
                                     V_tr.d_v_dense, V_tr.tile_cols[i],
                                     &beta,
                                     d_T,
                                     V_tr.tile_cols[i]));


        }
        CHECK_CUSPARSE(cusparseDestroySpMat(loc_V));

#ifdef DEBUG2D
        if (grid->world_rank==0)
        {
            std::cout<<"spmm "<<i<<std::endl;
        }
#endif

#ifndef BASIC
        timer.e_spmm += get_time_elapsed(comp_start);
#endif


        // Reduce along process columns into the ith rank
#ifndef BASIC
        auto reduce_start = hrc::now();
#endif

        MPI_Reduce(d_T, E.mat->dM, V.tile_rows[i] * E.mat->w_, MPI_FLOAT, MPI_SUM, i, grid->col_comm);

#ifdef DEBUG2D
        if (grid->world_rank==0)
        {
            std::cout<<"reduce "<<i<<std::endl;
        }
#endif
                    

#ifndef BASIC
        timer.e_mpi += get_time_elapsed(reduce_start);
#endif
        CHECK_CUDA(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);

    }

    delete[] recv_nnz;
    return EXIT_SUCCESS;
}


void rownnz_to_rowptrs(int * d_rowptrs, const int nrows)
{
    void * d_tmp = NULL;
    size_t tmp_size = 0;
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_tmp, tmp_size, d_rowptrs+1, nrows));
    CHECK_CUDA(cudaMalloc(&d_tmp, tmp_size));
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_tmp, tmp_size, d_rowptrs+1, nrows));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaDeviceSynchronize());
}


void rowptrs_to_rownnz(int * d_rowptrs, const int nrows)
{
    void * d_tmp = NULL;
    size_t tmp_size = 0;
    CHECK_CUDA(cub::DeviceAdjacentDifference::SubtractLeft(d_tmp, tmp_size, d_rowptrs, nrows+1, DiffOp<int>{}));
    CHECK_CUDA(cudaMalloc(&d_tmp, tmp_size));
    CHECK_CUDA(cub::DeviceAdjacentDifference::SubtractLeft(d_tmp, tmp_size, d_rowptrs, nrows+1, DiffOp<int>{}));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaDeviceSynchronize());
}


int spmm2d_bs_allgatherv(Handle& handle, DistV2D& V, DistV2D& V_tr, DistDnMat_t& K, DistDnMat_t& E, float * d_T)
{

    auto grid = V.grid;
    const int niters = grid->row_size;


    cusparseDnMatDescr_t T;

    // Transpose the sparse matrix
    int tr_rank = (grid->col_rank) * grid->col_size + grid->row_rank;
    V_tr.nnz = V.tile_nnz[tr_rank];

#ifndef BASIC
    auto preprocess = hrc::now();
#endif

    MPI_Sendrecv(V.d_vals, V.nnz, MPI_FLOAT, tr_rank, 100, 
                 V_tr.d_vals, V_tr.nnz, MPI_FLOAT, tr_rank, 100,
                 grid->world_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(V.d_rowinds, V.nnz, MPI_INT, tr_rank, 101, 
                 V_tr.d_rowinds, V_tr.nnz, MPI_INT, tr_rank, 101,
                 grid->world_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(V.d_colptrs, V.cols+1, MPI_INT, tr_rank, 102, 
                 V_tr.d_colptrs, V_tr.cols+1, MPI_INT, tr_rank, 102,
                 grid->world_comm, MPI_STATUS_IGNORE);


    // Allgatherv setup
    std::vector<int> recv_nnz(niters, 0);
    recv_nnz[grid->row_rank] = V_tr.nnz;
    MPI_Allreduce(MPI_IN_PLACE, recv_nnz.data(), niters, MPI_INT, MPI_SUM, grid->row_comm);

    std::vector<int> displs(niters, 0);
    std::exclusive_scan(recv_nnz.begin(), recv_nnz.end(), displs.begin(), 0);


    // Convert to CSR
    size_t buf_size = 0;
    void * d_buf = nullptr;
    CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(handle.sh(),
                                                 V_tr.cols,
                                                 V_tr.rows,
                                                 V_tr.nnz,
                                                 V_tr.d_vals,
                                                 V_tr.d_colptrs,
                                                 V_tr.d_rowinds,
                                                 V_tr.d_csr_val,
                                                 V_tr.d_csr_rowptrs,
                                                 V_tr.d_csr_colinds,
                                                 CUDA_R_32F,
                                                 CUSPARSE_ACTION_NUMERIC,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                                 &buf_size));
    CHECK_CUDA(cudaMalloc(&d_buf, buf_size));
    CHECK_CUSPARSE(cusparseCsr2cscEx2(handle.sh(),
                                     V_tr.cols,
                                     V_tr.rows,
                                     V_tr.nnz,
                                     V_tr.d_vals,
                                     V_tr.d_colptrs,
                                     V_tr.d_rowinds,
                                     V_tr.d_csr_val,
                                     V_tr.d_csr_rowptrs,
                                     V_tr.d_csr_colinds,
                                     CUDA_R_32F,
                                     CUSPARSE_ACTION_NUMERIC,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                     d_buf));

    CHECK_CUDA(cudaFree(d_buf));

    rowptrs_to_rownnz(V_tr.d_csr_rowptrs, V.rows);

#ifndef BASIC
    timer.e_other += get_time_elapsed(preprocess);
    auto mpi_start = hrc::now();
#endif


    CHECK_CUDA(cudaMemset(V_tr.d_remote_colptrs,0,sizeof(int)*(V_tr.global_rows + 1)));

    // Allgatherv csr arrays
    float * d_remote_vals = V_tr.d_remote_vals;
    int * d_remote_colinds = V_tr.d_remote_rowinds;
    int * d_remote_rowptrs = V_tr.d_remote_colptrs;
    MPI_Allgatherv(V_tr.d_csr_val, V_tr.nnz, MPI_FLOAT,
                   d_remote_vals, recv_nnz.data(), displs.data(),
                   MPI_FLOAT, grid->row_comm);
    MPI_Allgatherv(V_tr.d_csr_colinds, V_tr.nnz, MPI_FLOAT,
                   d_remote_colinds, recv_nnz.data(), displs.data(),
                   MPI_FLOAT, grid->row_comm);
    MPI_Allgather(V_tr.d_csr_rowptrs + 1, V.rows, MPI_INT,
                  d_remote_rowptrs + 1, V.rows, MPI_INT,
                  grid->row_comm);

    rownnz_to_rowptrs(d_remote_rowptrs, V_tr.global_rows);

#ifndef BASIC
    timer.e_mpi += get_time_elapsed(mpi_start);
    auto spmm_start = hrc::now();
#endif

    //print_device_matrix(d_remote_colinds, 1, V_tr.cols);
    //print_device_matrix(d_remote_vals, 1, V_tr.cols);
    //print_device_matrix(d_remote_rowptrs, 1, V_tr.global_rows + 1);

    // Build local csc
    cusparseSpMatDescr_t loc_V;
    CHECK_CUSPARSE(cusparseCreateCsr(&loc_V,
                                     V.global_rows,
                                     V_tr.cols,
                                     V_tr.cols,
                                     d_remote_rowptrs,
                                     d_remote_colinds,
                                     d_remote_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));
    float alpha = 1.0f;
    float beta = 0.0f;
    if (handle.isSparse())
    {
        CHECK_CUSPARSE(cusparseCreateDnMat(&T,
                                           V.global_rows,
                                           V_tr.cols,
                                           V_tr.cols,
                                           d_T,
                                           CUDA_R_32F,
                                           CUSPARSE_ORDER_ROW));
        // Buffer size 
        size_t buffer_size;
        void* buffer;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, loc_V, K.mat->M, &beta, T,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
        CHECK_CUDA(cudaMalloc(&buffer, buffer_size));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Perform SpMM
        CHECK_CUSPARSE(cusparseSpMM(
            handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, loc_V, K.mat->M, &beta, T,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Clean up
        CHECK_CUDA(cudaFree(buffer));
        CHECK_CUSPARSE(cusparseDestroyDnMat(T));

    }
    else
    {
        cusparseDnMatDescr_t v_dense;
        size_t buf_size = 0;
        void * d_buf = nullptr;
        CHECK_CUSPARSE(cusparseCreateDnMat(&v_dense, V.global_rows, V_tr.cols, V_tr.cols, V_tr.d_v_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(handle.sh(), loc_V, v_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &buf_size));
        CHECK_CUDA(cudaMalloc(&d_buf, buf_size));
        CHECK_CUSPARSE(cusparseSparseToDense(handle.sh(), loc_V, v_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_buf));
        CHECK_CUDA(cudaDeviceSynchronize());


        CHECK_CUDA(cudaFree(d_buf));
        CHECK_CUSPARSE(cusparseDestroyDnMat(v_dense));
        CHECK_CUBLAS(cublasSgemm(handle.dh(), 
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 V_tr.cols, V.global_rows, V_tr.cols,
                                 &alpha, 
                                 K.mat->dM, V_tr.cols,
                                 V_tr.d_v_dense, V_tr.cols,
                                 &beta,
                                 d_T,
                                 V_tr.cols));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(loc_V));

#ifndef BASIC
    timer.e_spmm += get_time_elapsed(spmm_start);
    mpi_start = hrc::now();
#endif

    // Reduce scatter final result into E
    int recvcount = E.mat->h_ * E.mat->w_;
    MPI_Reduce_scatter_block(d_T, E.mat->dM, recvcount, MPI_FLOAT, MPI_SUM, grid->col_comm);
                                     
#ifndef BASIC
    timer.e_mpi += get_time_elapsed(mpi_start);
#endif

    return EXIT_SUCCESS;
}

/*
 * Options:
 *  1. Allgatherv along rows -- reduce scatter along columns
 *  2. sqrt(P) bcasts along rows -- reduce scatter along columns 
 *  3. Allgather along rows, then -- reduce + scatter along columns
 *  4. Allgather along rows, then -- alltoallv + local reduction along columns
 */
int spmm15d(Handle& handle, DistV1D& V, DistDnMat_t& K, DistDnMat_t& E, DistDnMat_t& E_p, float * d_tmp, float * d_tmp2)
{
    auto grid2d = K.grid;
    auto grid2dcolmaj = E_p.grid;
    auto grid1d = V.grid;
    int sqrtp = grid2d->row_size;
    int p = grid2d->world_size;

    V_t * loc_v = V.local_v;
    DnMat_t * loc_k = K.mat;
    DnMat_t * loc_e = E.mat;
    DnMat_t * loc_e_p = E_p.mat;


    // Allgather local V along process rows of grid2d
    // We only have to communicate the rowinds array
    // Other stuff can be implicitly determined
    
#ifndef BASIC
    auto e_other_start = hrc::now();
#endif
    
    int * d_rowinds = V.d_remote_rowinds;

    if (grid2dcolmaj->row_rank == grid2dcolmaj->col_rank)
    {
        cudaMemcpy(d_rowinds + grid2dcolmaj->row_rank*loc_v->t, loc_v->local_ptr_to_assignments,
                    sizeof(float) * loc_v->t,
                    cudaMemcpyDeviceToDevice);
    }

#ifndef BASIC
    timer.e_other += get_time_elapsed(e_other_start);
    auto e_gather_start = hrc::now();
#endif

    MPI_Gather(loc_v->local_ptr_to_assignments, loc_v->t, MPI_INT,
                d_rowinds, loc_v->t, MPI_INT,
                grid2dcolmaj->row_rank, grid2dcolmaj->col_comm);
    MPI_Bcast(d_rowinds, loc_v->t*sqrtp, MPI_INT, grid2dcolmaj->col_rank, grid2dcolmaj->row_comm);

#ifndef BASIC
    timer.e_gather += get_time_elapsed(e_gather_start);
    e_other_start = hrc::now();
#endif

    float * d_vals = V.d_remote_vals;
    int * d_colptrs = V.d_remote_colptrs;
    CHECK_CUDA(cudaMemset(d_colptrs,0,sizeof(int)*(sqrtp*loc_v->t + 1)));

#ifndef BASIC
    auto e_spmm_start = hrc::now();
#endif

    // Set the values and colptrs arrays
    launch_init_from_rowinds_kernel(d_rowinds, d_colptrs, loc_v->global_cluster_sizes, d_vals, sqrtp * loc_v->t, sqrtp * loc_v->t);
    CHECK_CUDA(cudaDeviceSynchronize());


    cusparseSpMatDescr_t v_gather;
    //CHECK_CUSPARSE(cusparseCreateCsc(&v_gather,
    //                                 loc_v->k_,
    //                                 sqrtp*loc_v->t,
    //                                 sqrtp*loc_v->t,
    //                                 d_colptrs,
    //                                 d_rowinds,
    //                                 d_vals,
    //                                 CUSPARSE_INDEX_32I,
    //                                 CUSPARSE_INDEX_32I,
    //                                 CUSPARSE_INDEX_BASE_ZERO,
    //                                 CUDA_R_32F));
    csc_to_csr(handle, &v_gather, 
               loc_v->k_, sqrtp*loc_v->t, sqrtp*loc_v->t,
               d_vals, d_rowinds, d_colptrs,
               loc_v->d_csr_val, loc_v->d_csr_colinds, loc_v->d_csr_rowptrs);
    float alpha = 1.0f;
    float beta = 0.0f;
    if (handle.isSparse())
    {
        

        size_t buffer_size;
        void* buffer;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, v_gather, loc_k->M, &beta, loc_e_p->M,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
        CHECK_CUDA(cudaMalloc(&buffer, buffer_size));

        // Perform SpMM
        CHECK_CUSPARSE(cusparseSpMM(
            handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, v_gather, loc_k->M, &beta, loc_e_p->M,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Clean up
        CHECK_CUDA(cudaFree(buffer));
    }
    else
    {
        cusparseDnMatDescr_t v_dense;
        size_t buf_size = 0;
        void * d_buf = nullptr;
        //CHECK_CUDA(cudaMemset(loc_v->d_v_dense, 0, sizeof(float) * loc_v->k_ * loc_v->t*sqrtp));
        CHECK_CUSPARSE(cusparseCreateDnMat(&v_dense, loc_v->k_, sqrtp*loc_v->t, sqrtp*loc_v->t, V.d_v_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW));

        CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(handle.sh(), v_gather, v_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &buf_size));
        CHECK_CUDA(cudaMalloc(&d_buf, buf_size));
        CHECK_CUSPARSE(cusparseSparseToDense(handle.sh(), v_gather, v_dense, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_buf));
        CHECK_CUDA(cudaDeviceSynchronize());


        CHECK_CUDA(cudaFree(d_buf));
        CHECK_CUSPARSE(cusparseDestroyDnMat(v_dense));
        CHECK_CUBLAS(cublasSgemm(handle.dh(), 
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 sqrtp*loc_v->t, loc_v->k_, sqrtp*loc_v->t,
                                 &alpha, 
                                 loc_k->dM, sqrtp*loc_v->t,
                                 V.d_v_dense, sqrtp*loc_v->t,
                                 &beta,
                                 loc_e_p->dM,
                                 sqrtp*loc_v->t));
        CHECK_CUDA(cudaDeviceSynchronize());

    }
    CHECK_CUSPARSE(cusparseDestroySpMat(v_gather));

#ifndef BASIC
    timer.e_spmm += get_time_elapsed(e_spmm_start);
#endif

#ifndef BASIC
    auto e_trans_start = hrc::now();
#endif

    // Transpose local E partial sum
    CHECK_CUBLAS(cublasSgeam(handle.dh(),
                             CUBLAS_OP_T, 
                             CUBLAS_OP_N,
                             loc_e_p->h_, loc_e_p->w_,
                             &alpha, loc_e_p->dM,
                             loc_e_p->w_,
                             &beta,
                             d_tmp, loc_e_p->h_,
                             d_tmp, loc_e_p->h_));
    CHECK_CUDA(cudaDeviceSynchronize());

#ifndef BASIC
    timer.e_transpose += get_time_elapsed(e_trans_start);
#endif


#ifndef BASIC
    auto e_reduce_start = hrc::now();
#endif

    MPI_Reduce_scatter_block(d_tmp, d_tmp2, loc_e->h_ * loc_e->w_, MPI_FLOAT, MPI_SUM, grid2dcolmaj->col_comm);

#ifndef BASIC
    timer.e_reduce += get_time_elapsed(e_reduce_start);
#endif

#ifndef BASIC
    e_trans_start = hrc::now();
#endif

    // Need another transpose so output is row major
    CHECK_CUBLAS(cublasSgeam(handle.dh(),
                             CUBLAS_OP_T, 
                             CUBLAS_OP_N,
                             loc_e->w_, loc_e->h_,
                             &alpha, d_tmp2,
                             loc_e->h_,
                             &beta,
                             loc_e->dM, loc_e->w_,
                             loc_e->dM, loc_e->w_));
    CHECK_CUDA(cudaDeviceSynchronize());

#ifndef BASIC
    timer.e_transpose += get_time_elapsed(e_trans_start);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}


int spmm(Handle& handle, V_t& V, DnMat_t& K, DnMat_t& E) {
  // Define constants
  float alpha = 1.0f;
  float beta = 0.0f;

#ifndef BASIC
  auto e_spmm_start = hrc::now();
#endif

  if (handle.isSparse()) {

    cusparseSpMatDescr_t v_csr;

    csc_to_csr(handle, &v_csr, 
               V.k_, V.m_, V.m_,
               V.values, V.global_assignments, V.global_csc_col_offsets,
               V.d_csr_val, V.d_csr_colinds, V.d_csr_rowptrs);


    // Allocate workspace buffer
    size_t buffer_size;
    void* buffer;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, v_csr, K.M, &beta, E.M,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
    CHECK_CUDA(cudaMalloc(&buffer, buffer_size));

    // Perform SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, v_csr,
                                K.M, &beta, E.M, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, buffer));

    // Clean up
    CHECK_CUDA(cudaFree(buffer));
  } else {
    int64_t t = V.t;
    int64_t m = V.m_;
    int64_t k = V.k_;
    CHECK_CUBLAS(cublasSgemm(handle.dh(), CUBLAS_OP_N, CUBLAS_OP_N, t, k, m,
                             &alpha, K.dM, t, V.values, m, &beta, E.dM, t));
  }

  cudaDeviceSynchronize();
#ifndef BASIC
  timer.e_spmm += get_time_elapsed(e_spmm_start);
#endif
  return EXIT_SUCCESS;
}

int compute_z(V_t& V, DnMat_t& E, DnVec_t& z) {
  // Because we're using CSC, the cluster assignments vector is exactly the
  // CSC row indices vector of V
  launch_z_kernel(z.size, z.dz, V.local_ptr_to_assignments, E.dM);
  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}

int compute_z2d(DistV2D& V, DistDnMat_t& E, DistDnVec_t& z) 
{
    launch_z_kernel2d(V.nnz, V.cols, z.vec->dz, V.d_rowinds, V.d_colptrs, E.mat->dM);
    cudaDeviceSynchronize();
    return EXIT_SUCCESS;
}

int spmv(Handle& handle, V_t& V, DnVec_t& z, DnVec_t& c) {
  float alpha = 1.0f;
  float beta = 0.0f;

  if (handle.isSparse() || V.sparse) {
    // allocate an external buffer if needed
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.lV, z.z, &beta,
        c.z, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute SpMV
    CHECK_CUSPARSE(cusparseSpMV(handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, V.lV, z.z, &beta, c.z, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // cleanup
    CHECK_CUDA(cudaFree(dBuffer));
  } else {
    int64_t t = V.t;
    int64_t k = V.k_;
    int64_t m = V.m_;
    CHECK_CUBLAS(cublasSgemv(handle.dh(), CUBLAS_OP_T, t, k, &alpha,
                             V.local_ptr_to_values, m, z.dz, 1, &beta, c.dz,
                             1));
  }

  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}


int spmv(Handle& handle, DistV2D& V, DnVec_t& z, DnVec_t& c) 
{
  float alpha = 1.0f;
  float beta = 0.0f;

  // allocate an external buffer if needed
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.csc_mat, z.z, &beta,
      c.z, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  // execute SpMV
  CHECK_CUSPARSE(cusparseSpMV(handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, V.csc_mat, z.z, &beta, c.z, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

  // cleanup
  CHECK_CUDA(cudaFree(dBuffer));

  CHECK_CUDA(cudaDeviceSynchronize());
  return EXIT_SUCCESS;
}


int sum_vec(DnVec_t& c, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, c.dz, c.size, MPI_FLOAT, MPI_SUM, comm);
  return EXIT_SUCCESS;
}


int sum_vec2d(DistDnVec_t& c) {
    MPI_Allreduce(MPI_IN_PLACE, c.vec->dz, c.vec->size, MPI_FLOAT, MPI_SUM, c.grid->row_comm);
    return EXIT_SUCCESS;
}

int compute_c(Handle& handle, V_t& V, DnVec_t& z, DnVec_t& c, MPI_Comm comm) {
  spmv(handle, V, z, c);  // SpMV: c = Vz using local V
  sum_vec(c, comm);       // Calculate global c by summing across ranks
  return EXIT_SUCCESS;
}

int argmin(DnMat_t& E, DnVec_t& c, V_t& V, bool ptr) {
  bool t = true;
  cudaMemcpy(V.converged, &t, sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemset(V.local_k_means_objective_score, 0, sizeof(float));
  cudaMemset(V.local_k_means_objective_delta, 0, sizeof(float));
  if (ptr)
  {
      launch_argmin_kernel(
          V.k_, V.t, E.dM, c.dz, V.local_ptr_to_assignments, V.local_cluster_sizes,
          V.converged, V.local_k_means_objective_score,
          V.local_k_means_objective_delta, V.prev_point_to_cluster_distances);
  }
  else
  {
      launch_argmin_kernel(
          V.k_, V.t, E.dM, c.dz, V.local_assignments, V.local_cluster_sizes,
          V.converged, V.local_k_means_objective_score,
          V.local_k_means_objective_delta, V.prev_point_to_cluster_distances);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  return EXIT_SUCCESS;
}

int argmin2d(DistDnMat_t& E, DistDnVec_t& c, DistV2D& V) 
{

  int offset = V.grid->col_rank * V.tile_rows[0];


  launch_argmin_kernel_simple(
      V.rows, V.cols, E.mat->dM, c.vec->dz, V.d_colptrs, V.d_cluster_sizes,
      V.d_minpairs, offset);
  cudaDeviceSynchronize();


  MPI_Allreduce(MPI_IN_PLACE, V.d_minpairs, V.cols, MPI_FLOAT_INT, MPI_MINLOC, V.grid->col_comm);


  return EXIT_SUCCESS;
}

int gather_assignments(DnMat_t& E, DnVec_t& c, V_t& V, int convergence) {
  int send_count = V.t;
  int* send_buffer = V.local_assignments;
  int* recv_sizes = V.t_sizes;

  int dead_process_count = 0;
  if (convergence) {
    if (convergence == 2)
      recv_sizes = (int*)malloc(V.n_procs * sizeof(int));

    // bool locally_converged;
    // cudaMemcpy(&locally_converged, V.converged, sizeof(bool),
    //            cudaMemcpyDeviceToHost);

    float K_MEANS_EPSILON = 1e-7f;

    float local_k_means_objective_score, local_k_means_objective_delta;
    cudaMemcpy(&local_k_means_objective_score, V.local_k_means_objective_score,
               sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&local_k_means_objective_delta, V.local_k_means_objective_delta,
               sizeof(float), cudaMemcpyDeviceToHost);
    float local_k_means_rel_difference =
        fabs(local_k_means_objective_delta) /
        fabs(V.previous_local_k_means_objective_score);
    V.previous_local_k_means_objective_score = local_k_means_objective_score;
    bool locally_converged = (local_k_means_rel_difference < K_MEANS_EPSILON);

    float global_k_means_objective_score = 0.0f;
    float global_k_means_objective_delta = 0.0f;
    MPI_Allreduce(&local_k_means_objective_score,
                  &global_k_means_objective_score, 1, MPI_FLOAT, MPI_SUM,
                  V.comm);
    MPI_Allreduce(&local_k_means_objective_delta,
                  &global_k_means_objective_delta, 1, MPI_FLOAT, MPI_SUM,
                  V.comm);
    float k_means_rel_difference =
        fabs(global_k_means_objective_delta) /
        fabs(V.previous_global_k_means_objective_score);
    V.previous_global_k_means_objective_score = global_k_means_objective_score;
    bool globally_converged = (k_means_rel_difference < K_MEANS_EPSILON);

    // int rank;
    // MPI_Comm_rank(V.comm, &rank);
    // std::cout << "Rank " << rank << " --- "
    //           << "Local Score: " << local_k_means_objective_score
    //           << ", Local Delta: " << local_k_means_objective_delta
    //           << ", Global Score: " << global_k_means_objective_score
    //           << ", Global Delta: " << global_k_means_objective_delta
    //           << ", Relative Difference: " << k_means_rel_difference
    //           << ", Locally Converged: "
    //           << (locally_converged ? "true" : "false")
    //           << ", Globally Converged: "
    //           << (globally_converged ? "true" : "false") << std::endl;

    if (convergence == 2) {
      if (locally_converged) {
        // this process has locally converged, so it can be removed from allgather
        send_count = 0;
        send_buffer = nullptr;
      }
      bool local_convergence_ptr[V.n_procs];
      MPI_Allgather(&locally_converged, 1, MPI_C_BOOL, local_convergence_ptr, 1,
                    MPI_C_BOOL, V.comm);
      for (int i = 0; i < V.n_procs; ++i) {
        if (local_convergence_ptr[i]) {
          dead_process_count++;
          recv_sizes[i] = 0;
        } else {
          recv_sizes[i] = V.t_sizes[i];
        }
      }
    }
    if (globally_converged) {
      dead_process_count = V.n_procs;  // all processes have converged
    }
    if (dead_process_count == V.n_procs)
      return dead_process_count;  // all processes are dead, quit
  }

  // this one always utilizes all processes, but only passes k-size vector
  MPI_Allreduce(V.local_cluster_sizes, V.global_cluster_sizes, V.k_, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
  // this one will be reduced to the relevant processes as it allgathers a large
  // dense vector of points (if process-exlusion-based-convergence is enabled)
  MPI_Allgatherv(send_buffer, send_count, MPI_INT, V.global_assignments,
                 recv_sizes, V.displs, MPI_INT, MPI_COMM_WORLD);

  if (convergence == 2)
    free(recv_sizes);
  return dead_process_count;
}

int set_V_from_assignments(DnMat_t& E, DnVec_t& c, V_t& V) {
  launch_reinit_kernel(V.values, V.global_assignments, V.global_cluster_sizes,
                       V.k_, V.m_, V.sparse);
  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}


int set_V_from_assignments2d(DistV2D& V) 
{

  // Annoying
  launch_mininds_kernel(V.d_minpairs, V.d_mininds, V.cols);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Compute cluster sizes
  void * d_tmp = nullptr;
  size_t tmp_size = 0;
  CHECK_CUDA(cub::DeviceHistogram::HistogramEven(
                  d_tmp, tmp_size,
                  V.d_mininds, V.d_cluster_sizes, V.global_rows+1,
                  0, (int)V.global_rows, (int)V.cols));
  CHECK_CUDA(cudaMalloc(&d_tmp, tmp_size));
  CHECK_CUDA(cub::DeviceHistogram::HistogramEven(
                  d_tmp, tmp_size,
                  V.d_mininds, V.d_cluster_sizes, V.global_rows+1,
                  0, (int)V.global_rows, (int)V.cols));
  CHECK_CUDA(cudaFree(d_tmp));
  CHECK_CUDA(cudaDeviceSynchronize());
  MPI_Allreduce(MPI_IN_PLACE, V.d_cluster_sizes, V.global_rows, MPI_INT, MPI_SUM, V.grid->row_comm);

  int lower = V.tile_rows[0]* V.grid->col_rank;
  int upper = lower + V.tile_rows[0];

  //print_device_matrix(V.d_mininds, 1, V.cols);

  // Which points belong to my chunk of the clusters
  IsMine op{lower, upper};

  int64_t nnz_next = launch_countif(V.d_mininds, V.cols, op);
  V.update_nnz(nnz_next);

  //CHECK_CUDA(cudaFree(V.d_vals));
  //CHECK_CUDA(cudaFree(V.d_rowinds));
  CHECK_CUDA(cudaMemset(V.d_colptrs, 0, sizeof(int) * (V.cols + 1)));
  //CHECK_CUDA(cudaMalloc(&(V.d_vals), sizeof(float) * nnz_next));
  //CHECK_CUDA(cudaMalloc(&(V.d_rowinds), sizeof(int) * nnz_next));

  launch_copyif(V.d_mininds, V.d_rowinds, V.cols, op);
  CHECK_CUDA(cudaDeviceSynchronize());

  launch_reinit_kernel2d(V.d_vals, V.d_rowinds, V.d_colptrs, 
                         V.d_mininds, V.d_cluster_sizes,
                         V.tile_rows[0], V.cols, V.nnz, 
                         op);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Finally, inclusive prefix scan sets colptrs
  //print_device_matrix(V.d_colptrs, 1, V.cols+1);
  launch_inclusive_scan(V.d_colptrs + 1, V.d_colptrs + 1, V.cols);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Reinit cusparse csc
  cusparseDestroySpMat(V.csc_mat);
  V.init_cusparse_csc();
  //print_device_matrix(V.d_vals, 1, V.nnz);
  //CHECK_CUDA(cudaDeviceSynchronize());
  //print_device_matrix(V.d_rowinds, 1, V.nnz);
  //CHECK_CUDA(cudaDeviceSynchronize());
  //print_device_matrix(V.d_colptrs, 1, V.cols+1);
  //CHECK_CUDA(cudaDeviceSynchronize());


  return EXIT_SUCCESS;
}


int set_V_from_assignments15d(DistV1D& V)
{

  V_t * loc_v = V.local_v;
  
  MPI_Allreduce(loc_v->local_cluster_sizes, loc_v->global_cluster_sizes, loc_v->k_, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);

  launch_init_from_rowinds_kernel(loc_v->local_ptr_to_assignments,
                                  loc_v->local_csc_col_offsets,
                                  loc_v->global_cluster_sizes,
                                  loc_v->local_ptr_to_values,
                                  loc_v->t, loc_v->t);
  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}

int reinit_V(DnMat_t& E, DnVec_t& c, V_t& V) {
  argmin(E, c, V);                     // Launch argmin kernel
  gather_assignments(E, c, V, false);  // Gather assignments and cluster sizes
  set_V_from_assignments(E, c, V);     // Launch reinit kernel
  return EXIT_SUCCESS;
}

float compute_cluster_score(DnMat_t& K, DnMat_t& E, DnVec_t& c, V_t& V) {
  float* local_scores = (float*)malloc(V.t * sizeof(float));
  float* _local_scores;
  cudaMalloc(&_local_scores, V.t * sizeof(float));

  int row_offset = V.t_sizes[0] * V.rank;
  launch_score_kernel(_local_scores, K.dM + (row_offset * V.t), E.dM, c.dz,
                      V.local_assignments, V.t);
  cudaMemcpy(local_scores, _local_scores, V.t * sizeof(float),
             cudaMemcpyDeviceToHost);

  float score, local_score = 0;
  for (int i = 0; i < V.t; ++i) {
    local_score += local_scores[i];
  }

  MPI_Allreduce(&local_score, &score, 1, MPI_FLOAT, MPI_SUM, V.comm);
  free(local_scores);
  cudaFree(_local_scores);
  return score;
}

}  // namespace cpop
