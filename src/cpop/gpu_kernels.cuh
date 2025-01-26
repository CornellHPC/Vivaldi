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

/**
 * @brief Computes the Z vector
 * 
 * @param t The leading dimension of the ET matrix, usually the tile size/width
 * @param z The t-size array representing this function's output (the z vector)
 * @param assignments Cluster assignments on this process
 * @param ET This process's ET partial/submatrix (ET is k-by-t in row-major)
 */
void launch_z_kernel(int64_t t, float* z, int64_t* assignments, float* ET);

}  // namespace cpop

#endif  // CPOP_GPU_KERNELS_CUH
