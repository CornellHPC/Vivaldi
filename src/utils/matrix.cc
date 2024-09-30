#include "matrix.hh"

void grid_size(int mpi_size, int *p_out, int *q_out) {
  int p, q;
  for (p = int(sqrt(mpi_size)); p > 0; --p) {
    q = int(mpi_size / p);
    if (p * q == mpi_size)
      break;
  }
  *p_out = p;
  *q_out = q;
}

slate::Options get_slate_opts() {
#ifdef CUDA
  return {{slate::Option::Target, slate::Target::Devices}};
#else
  return {{slate::Option::Target, slate::Target::HostTask}};
#endif
}