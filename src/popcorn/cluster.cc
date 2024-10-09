#include "cluster.hh"

#include "mat/dense_mat.hh"
#include "utils/utils.hh"
#include "kernel/linear_kernel.cuh"

namespace popcorn {

void cluster(char *points_path, int m, int n, int k, MPI_Comm comm) {
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

  if (rank == 0) std::cout << "Reading data from " << points_path << std::endl;

  int p = square_grid_dim(comm);
  int mb = tile_dim(comm, m);
  int nb = tile_dim(comm, n);
  auto sP = DenseMat::load_from_file(points_path, m, n, mb, nb, p, comm);
  sP.print(std::cout, "P is");

  auto sB = sP.symmetric_product();
  // TODO: make gamma, c, r as IO input
  auto poly_kernel = PolynomialKernel(mb, 1.0f, 1.0f, 2.0f);
  sB.apply(poly_kernel);



  // TODO: make gamma, c, r as IO input
//   auto sK =
//       matrix::slate_point_mat_to_polynomial_kernel_mat(sP, 1.0f, 1.0f, 2.0f);
//   slate::print("K is ", sK);

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

}  // namespace popcorn
