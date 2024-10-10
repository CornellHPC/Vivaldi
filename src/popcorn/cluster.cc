#include "cluster.hh"

#include "kernel/linear_kernel.cuh"
#include "mat/dense_mat.hh"
#include "utils/utils.hh"

namespace popcorn {

void cluster(char *data_path, int m, int n, int k, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == 0)
#ifdef CUDA
    std::cout << "Running on: CUDA" << std::endl;
  wake_gpus(rank);
#else
    std::cout << "CUDA is unavailable. Some things may not work properly."
              << std::endl;
#endif

  if (rank == 0)
    std::cout << "Reading data from " << data_path << std::endl;

  auto P = DenseMat::load_from_file(data_path, m, n, comm);
  P.print("P");

  auto PT = P.transpose();
  PT.print("PT");

  auto B = P.gemm(PT);
  B.print("B");

  // TODO: make gamma, c, r as IO input
  // TODO: pull mb from the matrix instead of passing
  int mb = tile_dim(comm, m);
  auto poly_kernel = PolynomialKernel(mb, 1.0f, 1.0f, 2.0f);
  B.apply(poly_kernel);
  B.print("K");

  // auto cK = matrix::slate_mat_to_combblas_dpm(sK);
  // cK.PrintToFile("out/K");

  // if (rank == 0) std::cout << "Wrote K to disc" << std::endl;

  // auto cV =
  //     matrix::initialize_combblas_v_matrix(cK.getgnrow(), k, sK.mpiComm());
  // cV.PrintInfo();

  // combblas::spmm_stats stats;
  // auto O =
  //     combblas::SpMM_sC<SR, int64_t, DATA_TYPE, DATA_TYPE, UDER>(cV, cK,
  //     stats);
  // O.PrintToFile("out/O");
}

} // namespace popcorn
