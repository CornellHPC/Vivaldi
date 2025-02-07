#include <cassert>
#include <cstring>
#include <iostream>

#include "cuda_runtime.h"
#include "mpi.h"
#include "mpio.h"

#include "cluster.hh"
#include "gpu_kernels.cuh"
#include "utils.hh"

namespace cpop {

L_t::L_t(int64_t m, int64_t t, int64_t k, int* t_sizes) {
  ga = (int64_t*)calloc(m, sizeof(int64_t));
  la = (int64_t*)calloc(t, sizeof(int64_t));
  gl = (int64_t*)calloc(k, sizeof(int64_t));
  ll = (int64_t*)calloc(k, sizeof(int64_t));
  m_ = m;
  t_ = t;
  k_ = k;
  t_sizes_ = t_sizes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

  int rr_start_cluster = 0;  // round robin
  for (int i = 0; i < rank; ++i)
    rr_start_cluster += t_sizes[i];
  rr_start_cluster %= k;

  for (int i = 0; i < t; ++i) {
    int cluster = (rr_start_cluster + i) % k;
    la[i] = cluster;
    ll[cluster]++;
  }
}

int L_t::gather_clusters() {
  memset(gl, 0, k_ * sizeof(int64_t));
  MPI_Allreduce(ll, gl, k_, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  return EXIT_SUCCESS;
}

int L_t::gather_assignments() {
  auto displs = (int*)calloc(n_procs, sizeof(int));
  for (int i = 1; i < n_procs; ++i) {
    displs[i] = displs[i - 1] + t_sizes_[i - 1];
  }
  memset(ga, 0, m_ * sizeof(int64_t));
  MPI_Allgatherv(la, t_, MPI_INT64_T, ga, t_sizes_, displs, MPI_INT64_T,
                 MPI_COMM_WORLD);
  return EXIT_SUCCESS;
}

int L_t::save(const char* path, MPI_Comm comm) {
  MPI_File fh;
  MPI_File_open(comm, path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                &fh);

  // Compute offset for rank
  int offset = 0;
  for (int i = 0; i < rank; ++i) {
    offset += t_sizes_[i];
  }

  // Write the data to disk
  MPI_File_write_at(fh, offset * sizeof(int64_t), la, t_sizes_[rank],
                    MPI_INT64_T, MPI_STATUS_IGNORE);

  MPI_File_close(&fh);
  return EXIT_SUCCESS;
}

L_t::~L_t() {
  free(ga);
  free(la);
  free(gl);
  free(ll);
}

V_t::V_t(int64_t m, int64_t t, int64_t k) {
  CHECK_CUDA(cudaMalloc(&csr_row_offsets, (k + 1) * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&csr_col_inds, m * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&dgV, m * sizeof(int64_t)));
  CHECK_CUSPARSE(cusparseCreateCsr(&gV, k, m, m, csr_row_offsets, csr_col_inds,
                                   dgV, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CHECK_CUDA(cudaMalloc(&csc_col_offsets, (t + 1) * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&csc_row_inds, t * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&dlV, t * sizeof(int64_t)));
  CHECK_CUSPARSE(cusparseCreateCsc(&lV, k, t, t, csc_col_offsets, csc_row_inds,
                                   dlV, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  local_csr_row_offsets = (int64_t*)calloc(k + 1, sizeof(int64_t));
  local_csr_col_inds = (int64_t*)calloc(m, sizeof(int64_t));
  local_gV = (float*)calloc(m, sizeof(float));
  local_csc_col_offsets = (int64_t*)calloc(t + 1, sizeof(int64_t));
  local_csc_row_inds = (int64_t*)calloc(t, sizeof(int64_t));
  local_lV = (float*)calloc(t, sizeof(float));
  cluster_loc_ptrs = (int64_t*)calloc(k, sizeof(int64_t));

  m_ = m;
  t_ = t;
  k_ = k;
}

int V_t::reset_local() {
  memset(local_csr_row_offsets, 0, (k_ + 1) * sizeof(int64_t));
  memset(local_csr_col_inds, 0, m_ * sizeof(int64_t));
  memset(local_gV, 0, m_ * sizeof(float));
  memset(local_csc_col_offsets, 0, (t_ + 1) * sizeof(int64_t));
  memset(local_csc_row_inds, 0, t_ * sizeof(int64_t));
  memset(local_lV, 0, t_ * sizeof(float));
  memset(cluster_loc_ptrs, 0, k_ * sizeof(int64_t));
  return EXIT_SUCCESS;
}

int V_t::cp_local() {
  CHECK_CUDA(cudaMemcpy(csr_row_offsets, local_csr_row_offsets,
                        (k_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(csr_col_inds, local_csr_col_inds, m_ * sizeof(int64_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dgV, local_gV, m_ * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(csc_col_offsets, local_csc_col_offsets,
                        (t_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(csc_row_inds, local_csc_row_inds, t_ * sizeof(int64_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dlV, local_lV, t_ * sizeof(float), cudaMemcpyHostToDevice));
  reset_local();
  return EXIT_SUCCESS;
}

V_t::~V_t() {
  CHECK_CUDA(cudaFree(csr_row_offsets));
  CHECK_CUDA(cudaFree(csr_col_inds));
  CHECK_CUDA(cudaFree(csc_col_offsets));
  CHECK_CUDA(cudaFree(csc_row_inds));
  CHECK_CUDA(cudaFree(dgV));
  CHECK_CUDA(cudaFree(dlV));
  CHECK_CUSPARSE(cusparseDestroySpMat(gV));
  CHECK_CUSPARSE(cusparseDestroySpMat(lV));

  free(local_csr_row_offsets);
  free(local_csr_col_inds);
  free(local_gV);
  free(local_csc_col_offsets);
  free(local_csc_row_inds);
  free(local_lV);
  free(cluster_loc_ptrs);
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
  float* m = (float*)malloc(h_ * w_ * sizeof(float));
  CHECK_CUDA(
      cudaMemcpy(m, dM, h_ * w_ * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < h_; ++i) {
    for (int j = 0; j < w_; ++j) {
      std::cout << m[i * w_ + j] << " ";
    }
    std::cout << std::endl;
  }

  free(m);
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
  float* v = (float*)malloc(size * sizeof(float));
  CHECK_CUDA(cudaMemcpy(v, dz, size * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; ++i) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;

  free(v);
  return EXIT_SUCCESS;
}

DnVec_t::~DnVec_t() {
  CHECK_CUDA(cudaFree(dz));
  CHECK_CUSPARSE(cusparseDestroyDnVec(z));
}

int spmm(cusparseHandle_t& handle, V_t& V, DnMat_t& K, DnMat_t& E) {
  // Define constants
  float alpha = 1.0f;
  float beta = 0.0f;

  // Allocate workspace buffer
  size_t buffer_size;
  void* buffer;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.gV, K.M, &beta, E.M,
      CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &buffer_size));
  CHECK_CUDA(cudaMalloc(&buffer, buffer_size));

  // // Preprocess (may not work with later cusparseSpMV_preprocess)
  // cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
  //                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, gV, K, &beta,
  //                         *ET, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);

  // Perform SpMM
  CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.gV,
                              K.M, &beta, E.M, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT, buffer));

  // Clean up
  CHECK_CUDA(cudaFree(buffer));
  return EXIT_SUCCESS;
}

int compute_z(V_t& V, DnMat_t& E, DnVec_t& z) {
  // Because we're using CSC, the cluster assignments vector is exactly the
  // CSC row indices vector of V
  launch_z_kernel(z.size, z.dz, V.csc_row_inds, E.dM);
  return EXIT_SUCCESS;
}

int spmv(cusparseHandle_t& handle, V_t& V, DnVec_t& z, DnVec_t& c) {
  float alpha = 1.0f;
  float beta = 0.0f;

  // allocate an external buffer if needed
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.lV, z.z, &beta, c.z,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  // execute SpMV
  CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              V.lV, z.z, &beta, c.z, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

  // cleanup
  CHECK_CUDA(cudaFree(dBuffer));
  return EXIT_SUCCESS;
}

int sum_vec(DnVec_t& c, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, c.dz, c.size, MPI_FLOAT, MPI_SUM, comm);
  return EXIT_SUCCESS;
}

int reinit_ell(L_t& ell, DnMat_t& E, DnVec_t& c) {
  Argmin* da;
  CHECK_CUDA(cudaMalloc(&da, E.w_ * sizeof(Argmin)));
  launch_argmin_kernel(E.h_, E.w_, E.dM, c.dz, da);

  Argmin* a = (Argmin*)malloc(E.w_ * sizeof(Argmin));
  CHECK_CUDA(cudaMemcpy(a, da, E.w_ * sizeof(Argmin), cudaMemcpyDeviceToHost));

  // Update local assignments
  memset(ell.ll, 0, ell.k_ * sizeof(int64_t));
  for (int i = 0; i < E.w_; ++i) {
    Argmin x = a[i];
    ell.la[i] = x.mni;
    ell.ll[x.mni]++;
  }

  free(a);
  CHECK_CUDA(cudaFree(da));
  return EXIT_SUCCESS;
}

int reinit_V(V_t& V, L_t& ell) {
  V.reset_local();
  ell.gather_assignments();
  ell.gather_clusters();

  // TODO: convert to kernel for speediness (also reduces number of cudaMemcpys)
  for (int i = 0; i < V.k_; ++i)
    V.local_csr_row_offsets[i + 1] = V.local_csr_row_offsets[i] + ell.gl[i];
  for (int64_t i = 0; i < V.m_; ++i) {
    int cluster = ell.ga[i];  // get the cluster for this point
    int offset =
        V.local_csr_row_offsets[cluster] + (V.cluster_loc_ptrs[cluster]++);
    V.local_gV[offset] = 1.0f / ell.gl[cluster];
    V.local_csr_col_inds[offset] = i;
  }
  std::memcpy(V.local_csc_row_inds, ell.la, V.t_ * sizeof(int64_t));
  for (int i = 0; i < V.t_; ++i) {
    V.local_csc_col_offsets[i + 1] =
        i + 1;  // this works for CSC since 1 point per column
    V.local_lV[i] = 1.0f / ell.gl[ell.la[i]];
  }
  V.cp_local();  // copies the local buffers to GPU
  return EXIT_SUCCESS;
}

}  // namespace cpop
