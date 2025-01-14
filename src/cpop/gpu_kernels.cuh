#ifndef CPOP_GPU_KERNELS_CUH
#define CPOP_GPU_KERNELS_CUH

namespace cpop {

/**
 * Applies polynomial kernel function to matrix B.
 * 
 * @param m Number of rows in B
 * @param n Number of cols in B
 * @param B Matrix to apply kernel function to
 * @param gamma
 * @param c
 * @param r
 */
void launch_polynomial_kernel(int64_t m, int64_t n, float* B, float gamma,
                              float c, float r);

}  // namespace cpop

#endif  // CPOP_GPU_KERNELS_CUH
