#include <cassert>
#include <fstream>
#include <sstream>

#include "mpi.h"

#include "cpop/cluster.hh"
#include "cpop/dist_v.hh"
#include "cpop/compute_kernel.hh"
#include "cpop/utils.hh"
#include "colors.h"

using namespace cpop;

float EPSILON = 0.01;

template <typename T>
void assert_buffer_equal_host(T* b0, T* b1, int64_t count, const std::string& logfile_path) {
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
}

template <typename T>
void assert_buffer_equal_device(T* m0, T* m1, int64_t count, const std::string& logfile_path) 
{
  float* b0 = (float*)malloc(count * sizeof(float));
  float* b1 = (float*)malloc(count * sizeof(float));
  cudaMemcpy(b0, m0, count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b1, m1, count * sizeof(float), cudaMemcpyDeviceToHost);
  assert_buffer_equal_host(b0, b1, count, logfile_path);
  free(b0);
  free(b1);
}

void check_assignments(DistV2D& Vdist, V_t& V, const std::string& logpath)
{
    int * h_correct = new int[V.m_];
    CHECK_CUDA(cudaMemcpy(h_correct, V.global_assignments, sizeof(int) * V.m_, cudaMemcpyDeviceToHost));

    int * h_local = new int[Vdist.cols];
    CHECK_CUDA(cudaMemcpy(h_local, Vdist.d_mininds, sizeof(int) * Vdist.cols,
                            cudaMemcpyDeviceToHost));

    int * h_computed = new int[V.m_];
    MPI_Gather(h_local, Vdist.cols, MPI_INT, h_computed, Vdist.cols, MPI_INT, 0, Vdist.grid->row_comm);

    if (Vdist.grid->world_rank==0)
    {
        assert_buffer_equal_host(h_correct, h_computed, V.m_, logpath);
    }


    delete[] h_correct;
    delete[] h_computed;
    delete[] h_local;
}


void check_c(DistDnVec_t& c, DnVec_t& ccorrect, const std::string& logpath)
{
    float * d_c_computed;
    CHECK_CUDA(cudaMalloc(&d_c_computed, sizeof(float) * ccorrect.size));
    MPI_Gather(c.vec->dz, c.vec->size, MPI_FLOAT, d_c_computed, c.vec->size, MPI_FLOAT, 0, c.grid->col_comm);
    if (c.grid->world_rank==0)
    {
        assert_buffer_equal_device(ccorrect.dz, d_c_computed, ccorrect.size, logpath);
    }
}


int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, world_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);

  // Create logfile path for this rank
  std::stringstream logfile_ss;
  logfile_ss << "test_2d_rank_" << rank << ".log";
  std::string logfile_path = logfile_ss.str();

  int grid_size = std::floor(std::sqrt(world_size));
  std::shared_ptr<ProcessGrid> grid2d = std::make_shared<ProcessGrid>(grid_size, grid_size, true);

  /** Const */
  bool s = true;
  int m = 90;
  int n = 8;
  int k = 32;

  DistV2D Vdist(m, k, grid2d);
  int tr_rank = (grid2d->col_rank) * grid2d->col_size + grid2d->row_rank;
  DistV2D Vdist_tr(m, k, Vdist.tile_rows[grid2d->row_rank], Vdist.cols, grid2d);
  V_t V(m, k, s, comm);
  int t = V.t;  // get this process tile size
  

  par_print("V rows: %lu\n", Vdist.rows);

  wake_gpus(rank);
  slate::gpu_aware_mpi(true);
  Handle handle(s);

  auto PT = load_matrix("../data/randi", m, n, comm);
  DnMat_t K1D(m, t, compute_kernel_matrix(PT, 1.0f, 1.0f, 1.0f));
  PT.releaseWorkspace();

  auto PT2D = load_matrix2d("../data/randi", m, n, comm);
  DistDnMat_t K2D({new DnMat_t(Vdist.cols, Vdist.cols, compute_kernel_matrix2d(handle, PT2D, 1.0f, 1.0f, 1.0f, false)),
                   grid2d});
  PT2D.releaseWorkspace();

  float * d_T_buf;
  CHECK_CUDA(cudaMalloc(&d_T_buf, sizeof(float ) * (Vdist.rows+1) * Vdist.cols));
  DistDnMat_t E({new DnMat_t(Vdist.rows, Vdist.cols), 
                 grid2d});
  DistDnMat_t T({new DnMat_t(Vdist_tr.rows, Vdist.cols, d_T_buf), 
                 grid2d});
  DistDnVec_t z({new DnVec_t(Vdist.cols), grid2d});
  DistDnVec_t c({new DnVec_t(Vdist.rows), grid2d});


  DnMat_t Ecorrect(k, t);
  DnVec_t zcorrect(t);
  DnVec_t ccorrect(k);


  constexpr int niters = 5;
  for (int iter=0; iter<niters; iter++)
  {
      print_phase("Iteration beginning");

      spmm(handle, V, K1D, Ecorrect);
      spmm2d_bs(handle, Vdist, Vdist_tr, K2D, E, d_T_buf);
      //spmm2d(handle, Vdist, K2D, E);
      //print_device_matrix(E.mat->dM, E.mat->h_, E.mat->w_);
      //sleep(1);
      //fflush(stdout);
      //print_device_matrix(Ecorrect.dM, Ecorrect.h_, Ecorrect.w_);
      print_phase("SpMM Done");
      MPI_Barrier(MPI_COMM_WORLD);


      compute_z(V, Ecorrect, zcorrect);
      compute_z2d(Vdist, E, z);

      //print_device_matrix(z.vec->dz, 1, z.vec->size);
      //print_device_matrix(zcorrect.dz, 1, zcorrect.size);

      spmv(handle, V, zcorrect, ccorrect);
      sum_vec(ccorrect, comm);
      //print_device_matrix(ccorrect.dz, 1, ccorrect.size); 
      MPI_Barrier(MPI_COMM_WORLD);


      spmv(handle, Vdist, *z.vec, *c.vec);
      sum_vec2d(c);
      //print_device_matrix(c.vec->dz, 1, c.vec->size); 
      MPI_Barrier(MPI_COMM_WORLD);
      //print_phase("SpMV Done");

      check_c(c, ccorrect, logfile_path);
      print_phase("c vector correct");

      reinit_V(Ecorrect, ccorrect, V);

      argmin2d(E, c, Vdist);  
      //print_phase("Argmin Done");

      set_V_from_assignments2d(Vdist);
      //print_phase("Reinit Done");

      //print_device_matrix(Vdist.d_mininds, 1, Vdist.cols);
      //print_device_matrix(Vdist.d_minpairs, 1, Vdist.cols);
      //print_device_matrix(Vdist.d_cluster_sizes, 1, Vdist.global_rows);

      //print_device_matrix(Vdist.d_rowinds, 1, Vdist.nnz);
      //print_device_matrix(Vdist.d_vals, 1, Vdist.nnz);

      check_assignments(Vdist, V, logfile_path);

      par_print("NNZ V(%d, %d): %zu\n", grid2d->col_rank, grid2d->row_rank, Vdist.nnz);

      print_phase("Iteration correct");
  }

  if (rank==0)
  {
      std::cout<<GREEN<<"TEST PASSED"<<RESET<<std::endl;
  }


  MPI_Finalize();
  return 0;
}
