#ifndef DISTRIBUTED_POPCORN_POLYNOMIAL_KERNEL
#define DISTRIBUTED_POPCORN_POLYNOMIAL_KERNEL

// Local imports
#include "../../const.hh"
#include "kernel.cuh"

namespace popcorn {

/**
 * @brief Computes the polynomial kernel in-place on B
 *
 * @param args: contains n, gamma, c, r
 */
struct PolynomialKernel : public Kernel {
  PolynomialKernel(DATA_TYPE gamma, DATA_TYPE c, DATA_TYPE r)
      : gamma(gamma), c(c), r(r) {};
  void f(int64_t m, int64_t n, DATA_TYPE *B);

  DATA_TYPE gamma;
  DATA_TYPE c;
  DATA_TYPE r;
};

} // namespace popcorn

#endif // DISTRIBUTED_POPCORN_POLYNOMIAL_KERNEL
