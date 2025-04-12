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
void launch_z_kernel(int64_t t, float* z, int* assignments, float* ET);

/**
 * @brief Launches the argmin kernel, which computes D = -2E + c followed by a
 * column-wise argmin to determine the new cluster assignments for each point.
 * 
 * @param k The number of clusters, i.e. the height of E
 * @param t The number of points in this tile, i.e. the width of E
 * @param dE The local E matrix of size k-by-t
 * @param dc The local c vector of size k
 * @param local_assignments The local cluster assignments vector of size t
 * @param local_cluster_sizes The local cluster sizes vector of size k
 * @param converged A boolean flag where the kernel will write false if not converged
 */
void launch_argmin_kernel(int64_t k, int64_t t, float* dE, float* dc,
                          int* local_assignments, int* local_cluster_sizes,
                          bool* converged, float* local_k_means_objective_score,
                          float* local_k_means_objective_delta,
                          float* prev_point_to_cluster_distances);

/**
 * @brief Launches the reinit kernel, which computes V = 1 / cluster_size
 * 
 * @param V_global_values Full m-length vector of V values
 * @param global_assignments Full m-length vector of point-to-cluster assignments
 * @param global_cluster_sizes K-length vector of global cluster sizes
 * @param k Number of clusters
 * @param m Number of points
 * @param sparse Whether or not V has sparse representation
 */
void launch_reinit_kernel(float* V_global_values, int* global_assignments,
                          int* global_cluster_sizes, int64_t k, int64_t m,
                          bool sparse);

/**
  * @brief Launches the cluster score kernel, which computes the sum of squared distance
  *
  * @param local_scores The output length t vector of individual point distances
  * @param dK Offset into K matrix
  * @param dE Pointer to E matrix
  * @param dc Pointer to c vector
  * @param local_assignments Pointer to cluster assignments
  * @param t The tile width
  */
void launch_score_kernel(float* local_scores, float* dK, float* dE, float* dc,
                         int* local_assignments, int64_t t);

/**
 * @brief Compares two assignments vectors for exact equality
 * 
 * @param assignments Assignments vector to compare
 * @param prev_assignments Previous assignments vector to compare
 * @param t Assignments vector length
 * @return true if the vectors are equal, false otherwise
 */
bool test_convergence_equality(int* assignments, int* prev_assignments,
                               int64_t t);

}  // namespace cpop

#endif  // CPOP_GPU_KERNELS_CUH
