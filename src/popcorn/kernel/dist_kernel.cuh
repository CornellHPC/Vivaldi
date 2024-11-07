#ifndef DISTRIBUTED_POPCORN_DIST_KERNEL
#define DISTRIBUTED_POPCORN_DIST_KERNEL

// Local imports
#include "../../const.hh"

namespace popcorn {

struct DistKernel {
  /**
   * @brief Distance Kernel calculation
   * 
   * @param m rows in ET
   * @param n cols in ET
   * @param ET ET partial (NOT ON GPU)
   * @param c c_norm partial (NOT ON GPU)
   * @returns D matrix on the host
   */
  DATA_TYPE* kernel(int64_t m, int64_t n, DATA_TYPE* ET_host, DATA_TYPE* c_host);
};

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_DIST_KERNEL
