#ifndef CPOP_GPU_KERNELS_CUH
#define CPOP_GPU_KERNELS_CUH

namespace cpop {

/**
 * @brief Attaches index information to minimum value
 */
struct Argmin {
  float mn;     // The minimum value
  int64_t mni;  // The index of the minimum value
};

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

/**
 * @brief Launches the argmin kernel, which computes D = -2E + c followed by a
 * column-wise argmin to determine the new cluster assignments for each point.
 * 
 * @param k The number of clusters, i.e. the height of E
 * @param t The number of points in this tile, i.e. the width of E
 * @param dE The local E matrix of size k-by-t
 * @param dc The local c vector of size k
 * @param a The local cluster assignments vector of size t
 */
void launch_argmin_kernel(int64_t k, int64_t t, float* dE, float* dc,
                          Argmin* a);

/**
 * @brief Launches the D kernel, which computes D = -2E + c, takes the argmin
 * over columns, and stores the result in a.
 * 
 * @param k The number of clusters, i.e. the height of E
 * @param t The number of points in this tile, i.e. the width of E
 * @param dE The local E matrix of size k-by-t
 * @param dc The local c vector of size k
 * @param a The local cluster assignments vector of size t
 */
void launch_d_kernel(int64_t k, int64_t t, float* dE, float* dc, float* a);

}  // namespace cpop

#endif  // CPOP_GPU_KERNELS_CUH
