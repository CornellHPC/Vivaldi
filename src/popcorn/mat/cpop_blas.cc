#include "cpop_blas.hh"

// Local imports
#include "../kernel/argmin_kernel.cuh"
#include "../kernel/dist_kernel.cuh"
#include "../utils/utils.hh"

c_sp_ptr popcorn::initialize_from_coo(std::vector<float>& row_ids,
                                      std::vector<float>& col_ids,
                                      std::vector<DATA_TYPE>& vals,
                                      int64_t rows, int64_t cols,
                                      MPI_Comm comm) {
  int grid_dim = square_grid_dim(comm);

  // Initialize CombBLAS communicator grid
  auto grid = std::make_shared<combblas::CommGrid>(comm, grid_dim, grid_dim);

  // Initialize distributed data vectors
  combblas::FullyDistVec<int64_t, DATA_TYPE> drows(row_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dcols(col_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dvals(vals, grid);

  // Initialize distributed sparse matrix
  return std::make_unique<combblas::SpParMat<int64_t, DATA_TYPE, UDER>>(
      rows, cols, drows, dcols, dvals, false);
}

c_sp_ptr popcorn::initialize_v(int64_t points, int64_t k, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  int grid_dim = square_grid_dim(comm);
  int num_processes = grid_dim * grid_dim;

  // Compute points per cluster assuming round-robin assignment
  int* points_per_cluster = (int*)calloc(sizeof(int), k);
  for (int i = 0; i < k; ++i) {
    points_per_cluster[i] = (points / k) + ((i < points % k) ? 1 : 0);
  }

  // Compute points per process assuming round-robin assignment
  int* points_per_process = (int*)calloc(sizeof(int), num_processes);
  for (int i = 0; i < num_processes; ++i) {
    points_per_process[i] =
        (points / num_processes) + ((i < points % num_processes) ? 1 : 0);
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

  auto V = initialize_from_coo(lrow_ids, lcol_ids, lvals, points, k, comm);
  V->Transpose();
  return V;
}

c_dn_ptr popcorn::spmm(c_sp_ptr& V, c_dn_ptr& K) {
  // Setup for SpMM
  auto O = std::make_unique<combblas::DnParMat<int64_t, DATA_TYPE>>();
  combblas::spmm_stats stats;  // TODO: use this?

  // Perform SpMM
  *O = combblas::SpMM_sC<SR>(*V, *K, stats);

  return O;
}

std::vector<DATA_TYPE> popcorn::initialize_cnorm(c_sp_ptr& V, c_dn_ptr& ET) {
  // TODO: opportunity to apply threading, especially since csc_gespmv_dense
  // is not particularly advanced or accelerated
  // alternatively, we can use cusparse SPMV to speed up acceleration in the
  // future

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ml = V->getlocalrows();
  int nl = V->getlocalcols();
  UDER* spSeq = V->seqptr();

  DATA_TYPE* zl = (DATA_TYPE*)calloc(nl, sizeof(DATA_TYPE));
  for (typename UDER::SpColIter colit = spSeq->begcol();
       colit != spSeq->endcol(); ++colit) {
    for (typename UDER::SpColIter::NzIter nzit = spSeq->begnz(colit);
         nzit != spSeq->endnz(colit); ++nzit) {
      int i = nzit.rowid();
      int j = colit.colid();
      zl[j] = ET->getarr().at(i * nl + j);
    }
  }

  DATA_TYPE* cl = (DATA_TYPE*)calloc(ml, sizeof(DATA_TYPE));
  combblas::csc_gespmv_dense<SR>(*spSeq, zl, cl);

  std::vector<DATA_TYPE> c(ml, 0);
  MPI_Comm row_world = V->getcommgrid()->GetRowWorld();
  MPI_Allreduce(cl, c.data(), ml, MPI_FLOAT, MPI_SUM, row_world);

  // Clean up
  free(zl);
  free(cl);

  return c;
}

void popcorn::compute_d(c_dn_ptr& ET, std::vector<DATA_TYPE> cnorm) {
  // Setup the distance kernel
  auto dist_kernel = DistKernel();
  dist_kernel.kernel(ET->getnrow(), ET->getncol(), ET->getarr().data(),
                     cnorm.data());
}

c_sp_ptr popcorn::reinitialize_v(c_sp_ptr& V, c_dn_ptr& D) {
  // Initialize parameters
  int64_t clusters = V->getnrow();
  int64_t points = V->getncol();
  int64_t clusters_loc = V->getlocalrows();
  int64_t points_loc = V->getlocalcols();
  auto grid = V->getcommgrid();

  // Clamp local dimension
  if (clusters_loc == 0 || points_loc == 0) {
    clusters_loc = 0;
    points_loc = 0;
  }

  // ---------------------- WARNING ----------------------
  // D is transposed so local submatrix is actually of size
  // kloc by mloc (i.e. clusters are rows and points are cols)
  // ---------------------- WARNING ----------------------

  // Get global offsets
  int64_t cluster_offset =
      grid->GetRankInProcCol() * (clusters / grid->GetGridRows());
  int64_t point_offset =
      grid->GetRankInProcRow() * (points / grid->GetGridCols());

  // Compute local argmin
  auto argmin_kernel = ArgminKernel();
  auto M = argmin_kernel.kernel(clusters_loc, points_loc, cluster_offset, D->getarr().data());

  // Pad to target
  int64_t points_tar = points / grid->GetGridCols();
  M = (Argmin*)realloc(M, sizeof(Argmin) * points_tar);
  for (int i = points_loc; i < points_tar; ++i) {
    M[i] = Argmin{INFINITY, 0};
  }

  // Perform column reduction
  auto gM = (Argmin*)calloc(sizeof(Argmin), points_tar);
  MPI_Reduce(M, gM, points_tar, MPI_FLOAT_INT, MPI_MINLOC, 0,
             grid->GetColWorld());

  // Prepare to construct sparse matrix
  std::vector<float> lrow_ids, lcol_ids;
  std::vector<DATA_TYPE> lvals;

  // Perform row reduction on top row
  if (grid->GetRankInProcCol() == 0) {
    auto c = (int*)calloc(clusters, sizeof(int));
    auto gc = (int*)calloc(clusters, sizeof(int));
    for (int i = 0; i < points_tar; ++i) {
      Argmin a = gM[i];
      c[a.index]++;
    }

    MPI_Allreduce(c, gc, clusters, MPI_INT, MPI_SUM, grid->GetRowWorld());
    for (int i = 0; i < points_tar; ++i) {
      Argmin a = gM[i];
      lrow_ids.push_back(a.index);
      lcol_ids.push_back(point_offset + i);
      lvals.push_back(1.0f / gc[a.index]);
    }

    free(c);
    free(gc);
  }

  auto out = initialize_from_coo(lrow_ids, lcol_ids, lvals, clusters, points,
                                 V->getcommgrid()->GetWorld());

  free(M);
  free(gM);
  return out;
}

void popcorn::save_assignments(c_sp_ptr& V, const char* filename) {
  // Compute global offsets
  int row_offset = V->getcommgrid()->GetRankInProcCol() *
                   (V->getnrow() / V->getcommgrid()->GetGridRows());
  int col_offset = V->getcommgrid()->GetRankInProcRow() *
                   (V->getncol() / V->getcommgrid()->GetGridCols());

  // Open the file
  MPI_File fh;
  MPI_File_open(V->getcommgrid()->GetWorld(), filename,
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

  // Write the assignments
  UDER* spSeq = V->seqptr();
  for (typename UDER::SpColIter colit = spSeq->begcol();
       colit != spSeq->endcol(); ++colit) {
    for (typename UDER::SpColIter::NzIter nzit = spSeq->begnz(colit);
         nzit != spSeq->endnz(colit); ++nzit) {
      int cluster = row_offset + nzit.rowid();
      int point = col_offset + colit.colid();

      // TODO: Optimize the writes to be sequential
      MPI_File_write_at(fh, sizeof(int) * point, &cluster, 1, MPI_INT,
                        MPI_STATUS_IGNORE);
    }
  }

  // Clean up
  MPI_File_close(&fh);
}
