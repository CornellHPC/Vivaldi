#ifndef DISTRIBUTED_POPCORN_LINEAR_KERNEL
#define DISTRIBUTED_POPCORN_LINEAR_KERNEL

#include "../../const.hh"

namespace popcorn {

struct Kernel {
  void f(DATA_TYPE *B);
};

/**
 * @brief Computes the polynomial kernel in-place on B
 *
 * @param B B matrix
 * @param args: contains n, gamma, c, r
 */
struct PolynomialKernel : public Kernel {
  PolynomialKernel(int64_t n, DATA_TYPE gamma, DATA_TYPE c, DATA_TYPE r)
      : mb(mb), gamma(gamma), c(c), r(r) {};
  void f(DATA_TYPE *B);

  int64_t mb;
  DATA_TYPE gamma;
  DATA_TYPE c;
  DATA_TYPE r;
};

} // namespace popcorn

#endif // DISTRIBUTED_POPCORN_LINEAR_KERNEL
