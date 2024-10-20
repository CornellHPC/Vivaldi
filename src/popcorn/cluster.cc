// Local imports
#include "cluster.hh"
#include "cluster_assignment.hh"
#include "kernel/polynomial_kernel.cuh"
#include "mat/dense_mat.hh"
#include "mat/sparse_mat.hh"
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

  // Cull points to make matrix evenly divisible
  int grid_dim = square_grid_dim(comm);
  m -= m % grid_dim;

  if (rank == 0)
    std::cout << "Reading data from " << data_path << std::endl;

  // Load the point matrix
  auto P = DenseMat::load_from_file(data_path, m, n, comm);
  P.print("P");

  // Make a transposed copy of point matrix
  auto PT = P.transpose();
  PT.print("PT");

  // Compute the K matrix
  // TODO: make gamma, c, r as IO input
  auto K = P.gemm(PT);
  auto poly_kernel = PolynomialKernel(1.0f, 1.0f, 1.0f);
  K.apply(poly_kernel);
  K.print("K");

  // Initialize the V matrix
  ClusterAssignment assignment = ClusterAssignment::round_robin(m, k);
  auto V = SparseMat::initialize_v(assignment, comm);
  V.print("V");

  // Perform SpMM(VK)
  auto ET = V.spmm(K);
  ET.print("ET");

  // Initialize the z vector
  auto z = DenseMat::initialize_z(assignment, ET);
  if (rank == 0) {
    std::cout << "Got Z vector: ";
    for (int i = 0; i < z.size(); ++i) {
      std::cout << z.at(i) << " ";
    }
    std::cout << std::endl;
  }

  // Compute the centroid norms
  // auto C = V.spmm(z);
  // C.print("C");

  // TODO: Compute the D matrix
  // TODO: Update V matrix
  // TODO: Iterate
}

} // namespace popcorn
