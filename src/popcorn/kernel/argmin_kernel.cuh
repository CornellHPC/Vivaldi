#ifndef DISTRIBUTED_POPCORN_ARGMIN_KERNEL
#define DISTRIBUTED_POPCORN_ARGMIN_KERNEL

// Local imports
#include "../../const.hh"

namespace popcorn {

struct Argmin {
  DATA_TYPE value;
  int64_t index;
};

struct ArgminKernel {
  /**
   * @brief Argmin Kernel calculation
   *
   * @param m rows in D
   * @param n rows in D
   * @param D_host D (NOT ON GPU)
   */
  Argmin* kernel(int64_t m, int64_t n, DATA_TYPE* D_host);
};

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_ARGMIN_KERNEL
