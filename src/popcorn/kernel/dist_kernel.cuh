#ifndef DISTRIBUTED_POPCORN_DIST_KERNEL
#define DISTRIBUTED_POPCORN_DIST_KERNEL

// Local imports
#include "../../const.hh"

namespace popcorn {

struct DistKernel {
  /**
   * @brief Distance Kernel calculation. ET is overwritten by the result D
   * 
   * @param m rows in ET
   * @param n cols in ET
   * @param ET ET partial (NOT ON GPU), will be overwritten
   * @param c c_norm partial (NOT ON GPU)
   */
  void kernel(int64_t m, int64_t n, DATA_TYPE* ET_host, DATA_TYPE* c_host);
};

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_DIST_KERNEL
