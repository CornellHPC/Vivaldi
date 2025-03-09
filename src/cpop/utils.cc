#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

#include "utils.hh"

namespace cpop {

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
      "r", po::value<float>()->default_value(1.0f),
      "r parameter for the polynomial kernel")(
      "output,o", po::value<std::string>(),
      "output path for cluster assignments, default to \"[path]_out\"")(
      "benchmark,b", po::value<std::string>(),
      "path for benchmarked times, default to \"[path]_time\"")(
      "niter", po::value<int>()->default_value(100), "number of iterations")(
      "convergence",
      "if set, the algorithm will check for convergence (but still stop before "
      "\"--niter\" iterations)")("basic",
                                 "if set, the algorithm will run in basic mode "
                                 "(no finer-grained timing or barriers)");

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

  if (vm.count("convergence"))
    convergence = true;
  else
    convergence = false;
  if (vm.count("basic"))
    basic = true;
  else
    basic = false;
  if (vm.count("benchmark"))
    benchmark = vm["benchmark"].as<std::string>();
  else if (basic)
    benchmark = path + "_basic_time";
  else
    benchmark = path + "_time";
  if (basic && convergence) {
    std::cerr
        << "Error: Basic mode and convergence testing cannot be used together\n"
        << std::endl;
    exit(EXIT_FAILURE);
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

void Timer::save_all(const char* path) {
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

}  // namespace cpop
