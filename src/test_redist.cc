
#include <cassert>
#include <fstream>
#include <sstream>

#include "mpi.h"

#include "cpop/cluster.hh"
#include "cpop/compute_kernel.hh"
#include "cpop/utils.hh"

using namespace cpop;

float EPSILON = 0.01;

template <typename T>
void assert_buffer_equal(T* b0, T* b1, int64_t count, const std::string& logfile_path) {
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
      std::cout << b0[i] << " " << b1[i] << std::endl;
      logfile << i << "\t" << b0[i] << "\t" << b1[i] << "\t" << std::abs(b0[i] - b1[i]) << std::endl;
    }
    logfile.close();
  }
  assert(all_eq && "Arrays not equal");
}

void check_k(DnMat_t& Kcorrect, DnMat_t& Kcomputed, int rank, const std::string& logfile_path) {
  int64_t count = Kcorrect.h_ * Kcorrect.w_;
  float* m = (float*)malloc(count * sizeof(float));
  float* m2 = (float*)malloc(count * sizeof(float));
  cudaMemcpy(m, Kcorrect.dM, count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(m2, Kcomputed.dM, count * sizeof(float), cudaMemcpyDeviceToHost);

  assert_buffer_equal(m, m2, count, logfile_path);


  free(m);
  free(m2);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::stringstream ss;
  ss << "test2d_rank_" << rank << ".log";
  std::string logfile_path = ss.str();

  std::ofstream logfile(logfile_path);
  logfile << "Test2D Log for Rank " << rank << std::endl;
  logfile << "========================" << std::endl;
  logfile.close();

  /** Const */
  bool s = true;
  int m = 128;
  int n = 8;
  int k = 2;

  V_t V(m, k, s, comm);
  int t = V.t;  // get this process tile size

  wake_gpus(rank);
  slate::gpu_aware_mpi(true);

  Handle handle(s);

  auto PT = load_matrix("../data/randi", m, n, comm);
  DnMat_t Kcorrect(m, t, compute_kernel_matrix(PT, 1.0f, 1.0f, 1.0f));
  PT.releaseWorkspace();

  auto PT2 = load_matrix2d("../data/randi", m, n, comm);
  DnMat_t Kcomputed(m, t, compute_kernel_matrix2d(handle, PT2, 1.0f, 1.0f, 1.0f, true));
  PT2.releaseWorkspace();

  check_k(Kcorrect, Kcomputed, rank, logfile_path);
  print_phase("K correct");


  MPI_Finalize();
  return 0;
}
