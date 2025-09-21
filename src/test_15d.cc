
#include <cassert>
#include <fstream>
#include <sstream>

#include "mpi.h"

#include "cpop/cluster.hh"
#include "cpop/dist_v.hh"
#include "cpop/compute_kernel.hh"
#include "cpop/utils.hh"

using namespace cpop;

float EPSILON = 0.01;

template <typename T>
void assert_buffer_equal(T* m0, T* m1, int64_t count, const std::string& logfile_path) {
  float* b0 = (float*)malloc(count * sizeof(float));
  float* b1 = (float*)malloc(count * sizeof(float));
  cudaMemcpy(b0, m0, count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b1, m1, count * sizeof(float), cudaMemcpyDeviceToHost);
  bool all_eq = true;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int i = 0; i < count; ++i) {
    if (std::abs(b0[i] - b1[i]) > EPSILON)
      all_eq = false;
  }
  if (!all_eq) {
    std::cout << "Buffer mismatch on rank "<<rank<<":" << std::endl;
    std::ofstream logfile(logfile_path, std::ios::app);
    logfile << "Buffer mismatch on rank " << rank << ":" << std::endl;
    logfile << "Index\tBuffer0\tBuffer1\tDifference" << std::endl;
    for (int i = 0; i < count; ++i) {
      logfile << i << "\t" << b0[i] << "\t" << b1[i] << "\t" << std::abs(b0[i] - b1[i]) << std::endl;
    }
    logfile << "==============" << std::endl;
    logfile.close();
  }
  assert(all_eq && "Arrays not equal");
  free(b0);
  free(b1);
}



void check_e(DnMat_t& Ecorrect, DnMat_t& Ecomputed, int rank, const std::string& logfile_path) {
  int64_t count = Ecomputed.h_ * Ecomputed.w_;
  assert_buffer_equal(Ecorrect.dM, Ecomputed.dM, count, logfile_path);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, world_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);

  // Create logfile path for this rank
  std::stringstream logfile_ss;
  logfile_ss << "test_15d_rank_" << rank << ".log";
  std::string logfile_path = logfile_ss.str();
  int grid_size = std::floor(std::sqrt(world_size));
  std::shared_ptr<ProcessGrid> grid2d = std::make_shared<ProcessGrid>(grid_size, grid_size, false);
  std::shared_ptr<ProcessGrid> grid2dcolmaj = std::make_shared<ProcessGrid>(grid_size, grid_size, true);
  std::shared_ptr<ProcessGrid> grid1d = std::make_shared<ProcessGrid>(world_size, 1, false);

  /** Const */
  bool s = true;
  int m = 128;
  int n = 8;
  int k = 16;

  DistV1D Vdist(m, k, true, grid1d);
  V_t V(m, k, s, comm);
  int t = V.t;  // get this process tile size

  wake_gpus(rank);
  slate::gpu_aware_mpi(true);
  Handle handle(s);

  auto PT = load_matrix("../data/randi", m, n, comm);

  DnMat_t K1D(m, t, compute_kernel_matrix(PT, 1.0f, 1.0f, 1.0f));
  PT.releaseWorkspace();

  auto PT2D = load_matrix2d("../data/randi", m, n, comm);
  DistDnMat_t K2D({new DnMat_t(Vdist.local_v->t*grid_size, Vdist.local_v->t*grid_size, 
                             compute_kernel_matrix2d(handle, 
                                                     PT2D, 
                                                     1.0f,
                                                     1.0f,
                                                     1.0f,
                                                     false)),
                             grid2dcolmaj});

  PT2D.releaseWorkspace();

  int64_t row_tile_size = tile_dim(grid2d->col_comm, Vdist.local_v->k_);
  DistDnMat_t E({new DnMat_t(Vdist.local_v->k_, Vdist.local_v->t), 
                 grid1d});
  DistDnMat_t E_p({new DnMat_t(Vdist.local_v->k_, K2D.mat->w_), 
                  grid2dcolmaj});
  DistDnVec_t z({new DnVec_t(Vdist.local_v->t), grid1d});
  DistDnVec_t c({new DnVec_t(Vdist.local_v->k_), grid1d});


  DnMat_t Ecorrect(k, t);
  DnVec_t zcorrect(t);
  DnVec_t ccorrect(k);

  float * d_tmp;
  CHECK_CUDA(cudaMalloc(&d_tmp, sizeof(float) * Vdist.local_v->k_ * E_p.mat->w_));
  float * d_tmp2;
  CHECK_CUDA(cudaMalloc(&d_tmp2, sizeof(float) * Vdist.local_v->k_ * E.mat->w_));

  spmm(handle, V, K1D, Ecorrect);
  spmm15d(handle, Vdist, K2D, E, E_p, d_tmp, d_tmp2);

  check_e(Ecorrect, *E.mat, rank, logfile_path);
  print_phase("E1 correct");


  compute_z(V, Ecorrect, zcorrect);
  compute_z(*Vdist.local_v, *E.mat, *z.vec);

  spmv(handle, V, zcorrect, ccorrect);
  sum_vec(ccorrect, comm);

  spmv(handle, *Vdist.local_v, *z.vec, *c.vec);
  sum_vec(*c.vec, comm);


  reinit_V(Ecorrect, ccorrect, V);
  argmin(*E.mat, *c.vec, *Vdist.local_v, true);  

  set_V_from_assignments15d(Vdist);
  spmm(handle, V, K1D, Ecorrect);
  spmm15d(handle, Vdist, K2D, E, E_p, d_tmp, d_tmp2);
  check_e(Ecorrect, *E.mat, rank, logfile_path);
  print_phase("E2 correct");


  cudaFree(d_tmp);
  cudaFree(d_tmp2);

  MPI_Finalize();
  return 0;
}
