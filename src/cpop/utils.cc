#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

#include "utils.hh"

namespace cpop {

Timer timer;

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
      "process-exclusion-based-convergence)")(
      "alg,a", po::value<std::string>(),
      "clustering algorithm to use (1d, 1dr, 15d, 2d)");

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

  alg = vm["alg"].as<std::string>();
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


void Timer::allgather_nnz_perproc() {
    MPI_Allgather(nnz_perproc, niter, MPI_INT64_T, global_nnz_perproc, niter, MPI_INT64_T, MPI_COMM_WORLD);
}


void Timer::save_all(const char* path, float score) {

  if (global_nnz_perproc != nullptr) {
    allgather_nnz_perproc();
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
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
  file << "K Redist: " << k_redist << std::endl;
  file << "VI: " << vi_elapsed << std::endl;
  file << "E: " << e_elapsed << std::endl;
  file << "E Transpose: " << e_transpose << std::endl;
  file << "E MPI: " << e_mpi << std::endl;
  file << "E Reduce: " << e_reduce << " ms" << std::endl;
  file << "E Gather: " << e_gather << " ms" << std::endl;
  file << "E SpMM: " << e_spmm << std::endl;
  file << "E other: " << e_other << std::endl;
  file << "Z: " << z_elapsed << std::endl;
  file << "C: " << c_elapsed << std::endl;
  file << "C MPI: " << c_mpi << std::endl;
  file << "C Computation: " << c_computation << std::endl;
  file << "VR Computation: " << vr_computation << std::endl;
  file << "Elapsed: " << elapsed << std::endl;
  file << "Iterations before convergence: " << niter << std::endl;
  // file << "Cluster score: " << score << std::endl;
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
  if (global_nnz_perproc != nullptr) {
    file << "NNZ per proc: " << std::endl;
    for (int i=0; i<niter; ++i) {
      file << "    Iteration "<<i<<std::endl;
      for (int j=0; j<size; ++j) {
        file << "        Process "<<j<<": "<<global_nnz_perproc[j * niter + i]<<std::endl;
      }
    }
  }
      
  file.close();  // Close the file
  std::cout << "-------------------" << std::endl;
  std::cout << "IO: " << io << " ms" << std::endl;
  std::cout << "K: " << k_elapsed << " ms" << std::endl;
  std::cout << "K Redist: " << k_redist << " ms" << std::endl;
  std::cout << "VI: " << vi_elapsed << " ms" << std::endl;
  std::cout << "E: " << e_elapsed << " ms" << std::endl;
  std::cout << "E Transpose: " << e_transpose << " ms" << std::endl;
  std::cout << "E MPI: " << e_mpi << " ms" << std::endl;
  std::cout << "E Reduce: " << e_reduce << " ms" << std::endl;
  std::cout << "E Gather: " << e_gather << " ms" << std::endl;
  std::cout << "E SpMM: " << e_spmm << " ms" << std::endl;
  std::cout << "E other: " << e_other << std::endl;
  std::cout << "Z: " << z_elapsed << " ms" << std::endl;
  std::cout << "C: " << c_elapsed << " ms" << std::endl;
  std::cout << "C MPI: " << c_mpi << " ms" << std::endl;
  std::cout << "C Computation: " << c_computation << " ms" << std::endl;
  std::cout << "VR Computation: " << vr_computation << " ms" << std::endl;
  std::cout << "Elapsed: " << elapsed << " ms" << std::endl;
  std::cout << "Iterations before convergence: " << niter << std::endl;
  // std::cout << "Cluster score: " << score << std::endl;
  std::cout << "Results saved to: " << path << std::endl;
  std::cout << "-------------------" << std::endl;
}


void Timer::save_allranks(const char* path, float score) {

  if (global_nnz_perproc != nullptr) {
    allgather_nnz_perproc();
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string fpath(path);
  fpath += "_rank" + std::to_string(rank);

  std::ofstream file(fpath);  // Open the file in write mode
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << fpath
              << std::endl;  // Print error message if file cannot be opened
    return;                  // Exit the function if file cannot be opened
  }

  file << "IO: " << io << std::endl;
  file << "K: " << k_elapsed << std::endl;
  file << "K Redist: " << k_redist << std::endl;
  file << "VI: " << vi_elapsed << std::endl;
  file << "E: " << e_elapsed << std::endl;
  file << "E Transpose: " << e_transpose << std::endl;
  file << "E MPI: " << e_mpi << std::endl;
  file << "E Reduce: " << e_reduce << " ms" << std::endl;
  file << "E Gather: " << e_gather << " ms" << std::endl;
  file << "E SpMM: " << e_spmm << std::endl;
  file << "E other: " << e_other << std::endl;
  file << "Z: " << z_elapsed << std::endl;
  file << "C: " << c_elapsed << std::endl;
  file << "C MPI: " << c_mpi << std::endl;
  file << "C Computation: " << c_computation << std::endl;
  file << "VR Computation: " << vr_computation << std::endl;
  file << "Elapsed: " << elapsed << std::endl;
  file << "Iterations before convergence: " << niter << std::endl;
  if (global_nnz_perproc != nullptr) {
    file << "NNZ per proc: " << std::endl;
    for (int i=0; i<niter; ++i) {
      file << "    Iteration "<<i<<std::endl;
      for (int j=0; j<size; ++j) {
        file << "        Process "<<j<<": "<<global_nnz_perproc[j * niter + i]<<std::endl;
      }
    }
  }
      
  file.close();  // Close the file
  
  // Only rank 0 prints times
  if (rank==0) {
    std::cout << "-------------------" << std::endl;
    std::cout << "IO: " << io << " ms" << std::endl;
    std::cout << "K: " << k_elapsed << " ms" << std::endl;
    std::cout << "K Redist: " << k_redist << " ms" << std::endl;
    std::cout << "VI: " << vi_elapsed << " ms" << std::endl;
    std::cout << "E: " << e_elapsed << " ms" << std::endl;
    std::cout << "E Transpose: " << e_transpose << " ms" << std::endl;
    std::cout << "E MPI: " << e_mpi << " ms" << std::endl;
    std::cout << "E Reduce: " << e_reduce << " ms" << std::endl;
    std::cout << "E Gather: " << e_gather << " ms" << std::endl;
    std::cout << "E SpMM: " << e_spmm << " ms" << std::endl;
    std::cout << "E other: " << e_other << std::endl;
    std::cout << "Z: " << z_elapsed << " ms" << std::endl;
    std::cout << "C: " << c_elapsed << " ms" << std::endl;
    std::cout << "C MPI: " << c_mpi << " ms" << std::endl;
    std::cout << "C Computation: " << c_computation << " ms" << std::endl;
    std::cout << "VR Computation: " << vr_computation << " ms" << std::endl;
    std::cout << "Elapsed: " << elapsed << " ms" << std::endl;
    std::cout << "Iterations before convergence: " << niter << std::endl;
    // std::cout << "Cluster score: " << score << std::endl;
    std::cout << "Results saved to: " << path << std::endl;
    std::cout << "-------------------" << std::endl;
  }
}

int mod(int a, int b) {
    int r = a % b;
    return (r < 0) ? r + b : r;
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

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  
  float * d_x;
  float result;
  CHECK_CUDA(cudaMalloc(&d_x, sizeof(float) * 4));
  CHECK_CUBLAS(cublasSnrm2(handle, 4, d_x, 1, &result));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUBLAS(cublasDestroy(handle));

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int target = (rank + 1) % size;
  int source = mod(rank-1, size);
  int recv;
  MPI_Sendrecv(&rank, 1, MPI_INT, target, target, &recv, 1, MPI_INT, source, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
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

//std::vector<std::vector<std::array<int, 2>>> compute_tile_sizes2d(int m, int nprocs)
//{
//  int nprocs_root = std::floor(std::sqrt(nprocs));
//
//  std::vector<std::vector<std::array<int, 2>>> result;
//  for (int i=0; i<nprocs_root; i++)
//  {
//      result.emplace_back(nprocs_root);
//  }
//
//  int t = tile_dim2(nprocs_root, m);//m / nprocs + (m > nprocs * nprocs && m % nprocs > 0);
//  for (int i = 0; i < nprocs_root; ++i) {
//    for (int j=0; j<nprocs_root; j++)
//    {
//        if (i == nprocs_root - 1)
//        {
//            result[i][j][0] = m - (nprocs_root - 1) * t;
//        }
//        else
//        {
//            result[i][j][0] = t;
//        }
//
//        if (j == nprocs_root - 1)
//        {
//            result[i][j][1] = m - (nprocs_root - 1) * t;
//        }
//        else
//        {
//            result[i][j][1] = t;
//        }
//    }
//  }
//  return result;
//}

int square_grid_dim(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return std::floor(std::sqrt(size));
}

bool is_square_grid(MPI_Comm comm) {
  int size, sr;
  MPI_Comm_size(comm, &size);
  sr = square_grid_dim(comm);
  return sr * sr == size;
}

int tile_dim(MPI_Comm comm, int x) {
  int p = square_grid_dim(comm);
  return (x / p) + ((x % p == 0) ? 0 : 1);
}

int tile_dim2(int p, int x) {
  return (x / p) + ((x % p == 0) ? 0 : 1);
}

void print_phase(const char* name) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "==========" << name << "==========" << std::endl;
  }
  sleep(1);
  MPI_Barrier(MPI_COMM_WORLD);
}

void print_line() {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "==========" << __LINE__ << "==========" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace cpop
