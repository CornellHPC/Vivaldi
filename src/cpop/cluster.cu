#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cuda_runtime.h"
#include "mpi.h"
#include "mpio.h"

#include "cluster.hh"
#include "utils.hh"
#include "gpu_kernels.cuh"

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
    this->local_v = new V_t(m, k, sparse, grid->world_comm);
    this->grid = grid;
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

int spmm2d(Handle& handle, DistV2D& V, DistDnMat_t& K, DistDnMat_t& E)
{

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

#ifdef DEBUG2D
        par_print("Iteration %d\n", i);
#endif

        if (i == grid->col_rank)
        {
            K_send = K.mat->dM;
        }
        else
        {
            K_send = K_recv;
        }

        MPI_Bcast(K_send, K.mat->h_ * K.mat->w_, MPI_FLOAT, i, grid->col_comm);

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
        
        MPI_Bcast(d_vals_send, recv_nnz[i], MPI_FLOAT, i, grid->row_comm);
        MPI_Bcast(d_rowinds_send, recv_nnz[i], MPI_INT, i, grid->row_comm);
        MPI_Bcast(d_colptrs_send, V.tile_cols[i] + 1, MPI_INT, i, grid->row_comm);

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

    assert(loc_v->sparse && "1.5D only works with sparse V for now");

    // Allgather local V along process rows of grid2d
    // We only have to communicate the rowinds array
    // Other stuff can be implicitly determined
    
    int * d_rowinds;
    CHECK_CUDA(cudaMalloc(&d_rowinds, sizeof(int) * sqrtp * loc_v->t));

    if (grid2dcolmaj->row_rank == grid2dcolmaj->col_rank)
    {
        cudaMemcpy(d_rowinds + grid2dcolmaj->row_rank*loc_v->t, loc_v->local_ptr_to_assignments,
                    sizeof(float) * loc_v->t,
                    cudaMemcpyDeviceToDevice);
    }

    MPI_Gather(loc_v->local_ptr_to_assignments, loc_v->t, MPI_INT,
                d_rowinds, loc_v->t, MPI_INT,
                grid2dcolmaj->row_rank, grid2dcolmaj->col_comm);
    MPI_Bcast(d_rowinds, loc_v->t*sqrtp, MPI_INT, grid2dcolmaj->col_rank, grid2dcolmaj->row_comm);


    float * d_vals;
    int * d_colptrs;
    CHECK_CUDA(cudaMalloc(&d_vals, sizeof(float) * sqrtp * loc_v->t));
    //TODO: Technically we can initialize only once and reuse d_colptrs
    CHECK_CUDA(cudaMalloc(&d_colptrs, sizeof(int) * (sqrtp * loc_v->t + 1)));
    CHECK_CUDA(cudaMemset(d_colptrs,0,sizeof(int)*(sqrtp*loc_v->t + 1)));


    // Set the values and colptrs arrays
    launch_init_from_rowinds_kernel(d_rowinds, d_colptrs, loc_v->global_cluster_sizes, d_vals, sqrtp * loc_v->t, sqrtp * loc_v->t);
    CHECK_CUDA(cudaDeviceSynchronize());


    cusparseSpMatDescr_t v_gather;
    CHECK_CUSPARSE(cusparseCreateCsc(&v_gather,
                                     loc_v->k_,
                                     sqrtp*loc_v->t,
                                     sqrtp*loc_v->t,
                                     d_colptrs,
                                     d_rowinds,
                                     d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    float alpha = 1.0f;
    float beta = 0.0f;
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
    CHECK_CUSPARSE(cusparseDestroySpMat(v_gather));
    CHECK_CUDA(cudaFree(d_vals));
    CHECK_CUDA(cudaFree(d_colptrs));
    CHECK_CUDA(cudaFree(d_rowinds));


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

    //CHECK_CUDA(cudaMemset(loc_e_p->dM, 0,sizeof(float) * loc_e_p->h_ * loc_e_p->w_))


    MPI_Reduce_scatter_block(d_tmp, d_tmp2, loc_e->h_ * loc_e->w_, MPI_FLOAT, MPI_SUM, grid2dcolmaj->col_comm);


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

    MPI_Barrier(MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}


int spmm(Handle& handle, V_t& V, DnMat_t& K, DnMat_t& E) {
  // Define constants
  float alpha = 1.0f;
  float beta = 0.0f;

  if (handle.isSparse()) {
    // Allocate workspace buffer
    size_t buffer_size;
    void* buffer;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.gV, K.M, &beta, E.M,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
    CHECK_CUDA(cudaMalloc(&buffer, buffer_size));

    // Perform SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.gV,
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

  if (handle.isSparse()) {
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

  launch_argmin_kernel_simple(
      V.rows, V.cols, E.mat->dM, c.vec->dz, V.d_colptrs, V.d_cluster_sizes,
      V.d_minpairs);
  cudaDeviceSynchronize();


  MPI_Allreduce(MPI_IN_PLACE, V.d_minpairs, V.cols, MPI_FLOAT_INT, MPI_MINLOC, V.grid->col_comm);

  MPI_Allreduce(MPI_IN_PLACE, V.d_cluster_sizes, V.global_rows, MPI_INT, MPI_SUM, V.grid->world_comm);


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

  int lower = V.tile_cols[0] * V.grid->row_rank;
  int upper = lower + V.cols;

  // Which points belong to my chunk of the clusters
  IsMine op{lower, upper};

  int64_t nnz_next = launch_countif(V.d_mininds, V.cols, op);
  V.nnz = nnz_next;

  //CHECK_CUDA(cudaFree(V.d_vals));
  //CHECK_CUDA(cudaFree(V.d_rowinds));
  CHECK_CUDA(cudaMemset(V.d_colptrs, 0, sizeof(int) * (V.cols + 1)));
  //CHECK_CUDA(cudaMalloc(&(V.d_vals), sizeof(float) * nnz_next));
  //CHECK_CUDA(cudaMalloc(&(V.d_rowinds), sizeof(int) * nnz_next));

  launch_copyif(V.d_mininds, V.d_rowinds, V.cols, op);

  launch_reinit_kernel2d(V.d_vals, V.d_rowinds, V.d_colptrs, 
                         V.d_mininds, V.d_cluster_sizes,
                         V.rows, V.nnz, V.cols, 
                         op);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Finally, inclusive prefix scan sets colptrs
  launch_inclusive_scan(V.d_colptrs + 1, V.d_colptrs + 1, V.cols);

  // Reinit cusparse csc
  cusparseDestroySpMat(V.csc_mat);
  V.init_cusparse_csc();


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
