#ifndef DISTRIBUTED_POPCORN_ARGMIN_KERNEL
#define DISTRIBUTED_POPCORN_ARGMIN_KERNEL

// Local imports
#include "../../const.hh"

namespace popcorn {

struct Argmin {
  DATA_TYPE value;
  int index;
};

struct ArgminKernel {
  /**
   * @brief Argmin Kernel calculation
   *
   * @param m rows in D
   * @param n rows in D
   * @param gm global row offset
   * @param D_host D (NOT ON GPU)
   */
  Argmin* kernel(int m, int n, int gm, DATA_TYPE* D_host);
};

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_ARGMIN_KERNEL
