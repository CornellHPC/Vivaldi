#ifndef DISTRIBUTED_POPCORN_LINEAR_KERNEL
#define DISTRIBUTED_POPCORN_LINEAR_KERNEL

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
template <typename T>
void polynomial_kernel(int64_t n, T *B, T gamma, T c, T r);

} // namespace kernel

#endif // DISTRIBUTED_POPCORN_LINEAR_KERNEL
