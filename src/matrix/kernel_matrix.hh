#ifndef DISTRIBUTED_POPCORN_KERNEL_MATRIX_H
#define DISTRIBUTED_POPCORN_KERNEL_MATRIX_H

#include "../common.hh"
#include "../kernel/linear_kernel.cuh"
#include "matrix.hh"

namespace matrix {

/**
 * @brief Computes the B matrix from the sP points matrix. Used in the kernel
 * functions.
 *
 * @param sP matrix of size k by d
 * @return allocated matrix, with size k by k
 */
slate::Matrix<DATA_TYPE> compute_b_mat(slate::Matrix<DATA_TYPE> &sP);

slate::Matrix<DATA_TYPE> slate_point_mat_to_polynomial_kernel_mat(
    slate::Matrix<DATA_TYPE> &sP, DATA_TYPE gamma, DATA_TYPE c, DATA_TYPE r);

} // namespace matrix

#endif // DISTRIBUTED_POPCORN_KERNEL_MATRIX_H
