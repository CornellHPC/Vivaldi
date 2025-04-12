#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

#include "utils.hh"

namespace cpop {

const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "Unknown cuBLAS error";
  }
}

ArgParse::ArgParse(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "path,i", po::value<std::string>(), "path to the input dataset")(
      "npoints,m", po::value<int>(), "number of points in the dataset")(
      "nfeatures,n", po::value<int>(), "number of features")(
      "nclusters,k", po::value<int>(), "number of clusters to form")(
      "sparse,s", po::value<bool>()->default_value(true), "use sparse v")(
      "gamma", po::value<float>()->default_value(1.0f),
      "gamma parameter for the polynomial kernel")(
      "c", po::value<float>()->default_value(1.0f),
      "c parameter for the polynomial kernel")(
      "r", po::value<float>()->default_value(2.0f),
      "r parameter for the polynomial kernel")(
      "output,o", po::value<std::string>(),
      "output path for cluster assignments, default to \"[path]_out\"")(
      "benchmark,b", po::value<std::string>(),
      "path for benchmarked times, default to \"[path]_time\"")(
      "niter", po::value<int>()->default_value(100), "number of iterations")(
      "convergence", po::value<int>()->default_value(0),
      "enable convergence check (1 for basic, 2 for "
      "process-exclusion-based-convergence)");

  // Parse command line arguments
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(EXIT_SUCCESS);
  }

  if (vm.count("path") && vm.count("npoints") && vm.count("nfeatures") &&
      vm.count("nclusters")) {
    // mandatory arguments
    path = vm["path"].as<std::string>();
    m = vm["npoints"].as<int>();
    n = vm["nfeatures"].as<int>();
    k = vm["nclusters"].as<int>();
  } else {
    std::cerr << "Error: Missing mandatory arguments (\"--path/-i\", \"-m\", "
                 "\"-n\", \"-k\")\n"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (vm.count("output"))
    output = vm["output"].as<std::string>();
  else
    output = path + "_out";
  s = vm["sparse"].as<bool>();
  gamma = vm["gamma"].as<float>();
  c = vm["c"].as<float>();
  r = vm["r"].as<float>();
  niter = vm["niter"].as<int>();

  convergence = vm["convergence"].as<int>();

  if (vm.count("benchmark")) {
    benchmark = vm["benchmark"].as<std::string>();
  } else {
#ifdef BASIC
    benchmark = path + "_basic_time";
#else
    benchmark = path + "_time";
#endif
  }
}

void Timer::save_elapsed(const char* path) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0)
    return;  // Only the root process will save the results

  std::ofstream file(path);  // Open the file in write mode
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << path
              << std::endl;  // Print error message if file cannot be opened
    return;                  // Exit the function if file cannot be opened
  }
  file << elapsed << std::endl;  // Write the elapsed time to the file
  file.close();                  // Close the file
  std::cout << "-------------------" << std::endl;
  std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;
  std::cout << "Results saved to: " << path << std::endl;
  std::cout << "-------------------" << std::endl;
}

void Timer::save_all(const char* path, float score) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0)
    return;  // Only the root process will save the results

  std::ofstream file(path);  // Open the file in write mode
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << path
              << std::endl;  // Print error message if file cannot be opened
    return;                  // Exit the function if file cannot be opened
  }
  file << "IO: " << io << std::endl;
  file << "K: " << k_elapsed << std::endl;
  file << "VI: " << vi_elapsed << std::endl;
  file << "E: " << e_elapsed << std::endl;
  file << "Z: " << z_elapsed << std::endl;
  file << "C: " << c_elapsed << std::endl;
  file << "C MPI: " << c_mpi << std::endl;
  file << "C Computation: " << c_computation << std::endl;
  file << "VR: " << vr_elapsed << std::endl;
  file << "VR MPI: " << vr_mpi << std::endl;
  file << "VR Computation: " << vr_computation << std::endl;
  file << "Elapsed: " << elapsed << std::endl;
  file << "Iterations before convergence: " << niter << std::endl;
  file << "Cluster score: " << score << std::endl;
  if (dead_proc_counts != nullptr) {
    file << "Dead process counts: ";
    for (int i = 0; i < niter; ++i) {
      file << dead_proc_counts[i];
      if (i < niter - 1)
        file << ", ";
    }
    file << std::endl;
  } else {
    file << "Dead process counts: Not recorded" << std::endl;
  }
  file.close();  // Close the file
  std::cout << "-------------------" << std::endl;
  std::cout << "IO: " << io << " ms" << std::endl;
  std::cout << "K: " << k_elapsed << " ms" << std::endl;
  std::cout << "VI: " << vi_elapsed << " ms" << std::endl;
  std::cout << "E: " << e_elapsed << " ms" << std::endl;
  std::cout << "Z: " << z_elapsed << " ms" << std::endl;
  std::cout << "C: " << c_elapsed << " ms" << std::endl;
  std::cout << "C MPI: " << c_mpi << " ms" << std::endl;
  std::cout << "C Computation: " << c_computation << " ms" << std::endl;
  std::cout << "VR: " << vr_elapsed << " ms" << std::endl;
  std::cout << "VR MPI: " << vr_mpi << " ms" << std::endl;
  std::cout << "VR Computation: " << vr_computation << " ms" << std::endl;
  std::cout << "Elapsed: " << elapsed << " ms" << std::endl;
  std::cout << "Iterations before convergence: " << niter << std::endl;
  std::cout << "Cluster score: " << score << std::endl;
  std::cout << "Results saved to: " << path << std::endl;
  std::cout << "-------------------" << std::endl;
}

void wake_gpus(int rank) {
  int ndevices;
  cudaGetDeviceCount(&ndevices);

  if (rank == 0) {
    std::cout << "Number of GPUs per node: " << ndevices << "\n" << std::flush;
    std::cout << "Waking the GPUs..." << std::flush;
  }

  for (int i = 0; i < ndevices; ++i)
    cudaSetDevice(i);

  if (rank == 0)
    std::cout << " DONE!\n" << std::flush;
}

int64_t get_time_elapsed(std::chrono::_V2::system_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now() - start)
      .count();
}

int* compute_tile_sizes(int m, int nprocs) {
  int t = m / nprocs + (m > nprocs * nprocs && m % nprocs > 0);
  int* t_sizes = (int*)calloc(nprocs, sizeof(int));
  for (int i = 0; i < nprocs - 1; ++i) {
    t_sizes[i] = t;
  }
  t_sizes[nprocs - 1] = m - (nprocs - 1) * t;
  return t_sizes;
}

}  // namespace cpop
