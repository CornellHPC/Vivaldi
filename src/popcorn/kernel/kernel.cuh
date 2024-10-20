#ifndef DISTRIBUTED_POPCORN_KERNEL
#define DISTRIBUTED_POPCORN_KERNEL

// Local imports
#include "../../const.hh"

namespace popcorn {

struct Kernel {
  virtual void f(int64_t m, int64_t n, DATA_TYPE *B);
};

} // namespace popcorn

#endif // DISTRIBUTED_POPCORN_KERNEL
