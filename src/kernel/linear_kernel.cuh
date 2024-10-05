#ifndef DISTRIBUTED_POPCORN_LINEAR_KERNEL
#define DISTRIBUTED_POPCORN_LINEAR_KERNEL

#include "../const.hh"

namespace kernel {

/**
 * @brief Computes the polynomial kernel for input B and stores the result in B
 * 
 * @param n number of points, e.g. width/height of B
 * @param B B matrix
 * @param gamma const
 * @param c const
 * @param r const
 */
void polynomial_kernel(int64_t n, DATA_TYPE* B, DATA_TYPE gamma, DATA_TYPE c, DATA_TYPE r);

}

#endif  // DISTRIBUTED_POPCORN_LINEAR_KERNEL